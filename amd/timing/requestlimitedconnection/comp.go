// Package requestlimitedconnection provides a zero-latency connection with
// one or two independent shared memory-request issue limits. Responses are
// never throttled.
package requestlimitedconnection

import (
	"fmt"

	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

// Spec configures a request-limited connection.
type Spec struct {
	Freq                   timing.Freq   `json:"freq"`
	RequestRateNumerator   int           `json:"request_rate_numerator"`
	RequestRateDenominator int           `json:"request_rate_denominator"`
	BurstRequests          int           `json:"burst_requests"`
	RequestFilter          RequestFilter `json:"request_filter"`
	// The optional secondary bucket lets one request class (such as writes)
	// obey an additional rate while still consuming the primary total budget.
	SecondaryRequestRateNumerator   int           `json:"secondary_request_rate_numerator"`
	SecondaryRequestRateDenominator int           `json:"secondary_request_rate_denominator"`
	SecondaryBurstRequests          int           `json:"secondary_burst_requests"`
	SecondaryRequestFilter          RequestFilter `json:"secondary_request_filter"`
}

// RequestFilter selects which memory request class consumes token credits.
// The zero value preserves the original behavior and limits every AccessReq.
type RequestFilter string

const (
	AllRequests   RequestFilter = ""
	ReadRequests  RequestFilter = "read"
	WriteRequests RequestFilter = "write"
)

// State stores arbitration and token-bucket state.
type State struct {
	NextPortID               int    `json:"next_port_id"`
	RequestCredit            int    `json:"request_credit"`
	LastCreditCycle          uint64 `json:"last_credit_cycle"`
	SecondaryRequestCredit   int    `json:"secondary_request_credit"`
	SecondaryLastCreditCycle uint64 `json:"secondary_last_credit_cycle"`
}

type ports struct {
	ports   []messaging.Port
	portMap map[messaging.RemotePort]int
}

func (p *ports) addPort(port messaging.Port) {
	p.ports = append(p.ports, port)
	p.portMap[port.AsRemote()] = len(p.ports) - 1
}

func (p *ports) getPortIndex(index int) messaging.Port {
	return p.ports[index]
}

func (p *ports) getPortByName(name messaging.RemotePort) messaging.Port {
	portIndex, found := p.portMap[name]
	if !found {
		panic(fmt.Sprintf("port %s not found", name))
	}

	return p.ports[portIndex]
}

func (p *ports) list() []messaging.Port {
	return p.ports
}

func (p *ports) len() int {
	return len(p.ports)
}

// Comp connects ports while sharing request budgets across all sources and
// destinations.
type Comp struct {
	*modeling.Component[Spec, State, modeling.None]
}

func (c *Comp) mw() *middleware {
	return c.Middlewares()[0].(*middleware)
}

// PlugIn connects a port.
func (c *Comp) PlugIn(port messaging.Port) {
	c.Lock()
	defer c.Unlock()

	c.mw().ports.addPort(port)
	port.SetConnection(c)
}

// Unplug is not supported.
func (c *Comp) Unplug(_ messaging.Port) {
	panic("not implemented")
}

// NotifyAvailable wakes senders after a destination gains buffer space.
func (c *Comp) NotifyAvailable(available messaging.Port) {
	for _, port := range c.mw().ports.list() {
		if port == available {
			continue
		}
		port.NotifyAvailable()
	}
	c.TickNow()
}

// NotifySend starts connection ticking after a port queues a message.
func (c *Comp) NotifySend() {
	c.TickNow()
}

type middleware struct {
	comp  *modeling.Component[Spec, State, modeling.None]
	ports ports
}

func (m *middleware) Tick() bool {
	numPorts := m.ports.len()
	if numPorts == 0 {
		return false
	}

	state := m.comp.State
	spec := m.comp.Spec()
	currentCycle := spec.Freq.Cycle(m.comp.CurrentTime())
	requestCredit, creditDenominator, unlimited := refillTokenBucket(
		state.RequestCredit, state.LastCreditCycle,
		spec.RequestRateNumerator, spec.RequestRateDenominator,
		spec.BurstRequests, currentCycle,
	)
	secondaryRequestCredit, secondaryCreditDenominator, secondaryUnlimited :=
		refillTokenBucket(
			state.SecondaryRequestCredit, state.SecondaryLastCreditCycle,
			spec.SecondaryRequestRateNumerator,
			spec.SecondaryRequestRateDenominator,
			spec.SecondaryBurstRequests, currentCycle,
		)
	if !unlimited {
		(&m.comp.State).LastCreditCycle = currentCycle
	}
	if !secondaryUnlimited {
		(&m.comp.State).SecondaryLastCreditCycle = currentCycle
	}
	madeProgress := false
	waitingForCredit := false

	for i := range numPorts {
		portID := (i + state.NextPortID) % numPorts
		port := m.ports.getPortIndex(portID)
		madeProgress = m.forwardMany(
			port, &requestCredit, creditDenominator, unlimited,
			&secondaryRequestCredit, secondaryCreditDenominator,
			secondaryUnlimited,
			&waitingForCredit,
		) || madeProgress
	}

	(&m.comp.State).NextPortID = (state.NextPortID + 1) % numPorts
	(&m.comp.State).RequestCredit = requestCredit
	(&m.comp.State).SecondaryRequestCredit = secondaryRequestCredit

	return madeProgress || waitingForCredit
}

func refillTokenBucket(
	current int,
	lastCycle uint64,
	rateNumerator, rateDenominator, burstRequests int,
	currentCycle uint64,
) (credit, denominator int, unlimited bool) {
	denominator = rateDenominator
	if denominator <= 0 {
		denominator = 1
	}
	if rateNumerator <= 0 {
		return current, denominator, true
	}

	capacity := burstRequests * denominator
	minCapacity := rateNumerator + denominator - 1
	if capacity < minCapacity {
		capacity = minCapacity
	}
	if capacity < denominator {
		capacity = denominator
	}
	elapsedCycles := currentCycle - lastCycle
	if elapsedCycles == 0 {
		elapsedCycles = 1
	}
	credit = refillRequestCredit(
		current, capacity, rateNumerator, elapsedCycles,
	)
	return credit, denominator, false
}

func refillRequestCredit(
	current, capacity, rateNumerator int,
	elapsedCycles uint64,
) int {
	if current >= capacity {
		return capacity
	}

	creditsNeeded := capacity - current
	cyclesToFull := uint64(
		(creditsNeeded + rateNumerator - 1) / rateNumerator,
	)
	if elapsedCycles >= cyclesToFull {
		return capacity
	}

	return current + int(elapsedCycles)*rateNumerator
}

func (m *middleware) forwardMany(
	port messaging.Port,
	requestCredit *int,
	creditDenominator int,
	unlimited bool,
	secondaryRequestCredit *int,
	secondaryCreditDenominator int,
	secondaryUnlimited bool,
	waitingForCredit *bool,
) bool {
	madeProgress := false

	for {
		head := port.PeekOutgoing()
		if head == nil {
			break
		}

		isRequest := requestConsumesCredit(head, m.comp.Spec().RequestFilter)
		isSecondaryRequest := requestConsumesCredit(
			head, m.comp.Spec().SecondaryRequestFilter,
		)
		primaryBlocked := isRequest && !unlimited &&
			*requestCredit < creditDenominator
		secondaryBlocked := isSecondaryRequest && !secondaryUnlimited &&
			*secondaryRequestCredit < secondaryCreditDenominator
		if primaryBlocked || secondaryBlocked {
			*waitingForCredit = true
			break
		}

		dstPort := m.ports.getPortByName(head.Meta().Dst)
		if !dstPort.CanDeliver() {
			break
		}

		dstPort.Deliver(head)
		port.RetrieveOutgoing()
		madeProgress = true

		if isRequest && !unlimited {
			*requestCredit -= creditDenominator
		}
		if isSecondaryRequest && !secondaryUnlimited {
			*secondaryRequestCredit -= secondaryCreditDenominator
		}
	}

	return madeProgress
}

func requestConsumesCredit(msg messaging.Msg, filter RequestFilter) bool {
	if _, ok := msg.(memprotocol.AccessReq); !ok {
		return false
	}

	switch filter {
	case ReadRequests:
		_, ok := msg.(memprotocol.ReadReq)
		return ok
	case WriteRequests:
		_, ok := msg.(memprotocol.WriteReq)
		return ok
	default:
		return true
	}
}
