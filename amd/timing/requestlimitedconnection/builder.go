package requestlimitedconnection

import (
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
)

var defaultSpec = Spec{
	Freq: 1 * timing.GHz,
}

// DefaultSpec returns the default unlimited connection configuration.
func DefaultSpec() Spec {
	return defaultSpec
}

// Builder constructs request-limited connections.
type Builder struct {
	spec      Spec
	registrar modeling.Registrar
}

// MakeBuilder creates a builder with an unlimited request rate.
func MakeBuilder() Builder {
	return Builder{spec: defaultSpec}
}

// WithRegistrar sets the simulation that owns the connection.
func (b Builder) WithRegistrar(registrar modeling.Registrar) Builder {
	b.registrar = registrar
	return b
}

// WithSpec sets the complete connection specification.
func (b Builder) WithSpec(spec Spec) Builder {
	b.spec = spec
	return b
}

// Build creates and registers a request-limited connection.
func (b Builder) Build(name string) *Comp {
	if b.registrar == nil {
		panic("requestlimitedconnection: WithRegistrar is required")
	}

	engine := b.registrar.GetEngine()
	modelComp := modeling.NewBuilder[Spec, State, modeling.None]().
		WithEngine(engine).
		WithFreq(b.spec.Freq).
		WithSpec(b.spec).
		Build(name)
	if b.spec.BurstRequests > 0 {
		denominator := b.spec.RequestRateDenominator
		if denominator <= 0 {
			denominator = 1
		}
		modelComp.State.RequestCredit = b.spec.BurstRequests * denominator
	}

	modelComp.TickingComponent = modeling.NewSecondaryTickingComponent(
		name, engine, b.spec.Freq, modelComp,
	)

	mw := &middleware{
		comp: modelComp,
		ports: ports{
			ports:   make([]messaging.Port, 0),
			portMap: make(map[messaging.RemotePort]int),
		},
	}
	modelComp.AddMiddleware(mw)

	connection := &Comp{Component: modelComp}
	b.registrar.RegisterConnection(connection)

	return connection
}
