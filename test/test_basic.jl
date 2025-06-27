using PeriodicOrbit
using NbodyGradient
using LinearAlgebra: dot, cross
using Test

function Base.isapprox(a::Elements,b::Elements;tol=1e-10)
    fields = setdiff(fieldnames(Elements),[:a,:e,:ϖ])
    for i in fields
        af = getfield(a,i)
        bf = getfield(b,i)
        if abs(af - bf) > tol
            return false
        end
    end
    return true
end

@testset "Cartesian to Elements" begin
    ωs = [0., π/2, π, -π/2, 1e-4, -1e-4, 
    π/2 + 1e-4, π/2 - 1e-4, 
    -π/2 + 1e-4, -π/2 - 1e-4, 
    π + 1e-4, π - 1e-4, 
    -π + 1e-4, -π - 1e-4]

    es = [0.0001, 0.1, 0.5, 0.9]

    # ωs = [0., π/2, π, -π/2, 1e-4, -1e-4]
    # es = [0.0001, 0.1, 0.5, 0.9]
    # Is = [0., 1., π/2]
    # Ωs = [0., 1., π/2, -π/2]

    for ω in ωs, e in es
        p1 = Elements(m=1)
        p2 = Elements(m=1e-4, P=365.242, e=e, ω=ω, I=π/2, Ω=0)
        p3 = Elements(m=1e-4, P=2*365.242, e=e, ω=ω, I=π/2, Ω=0)


        ic = ElementsIC(0., 3, p1, p2, p3)
        s = State(ic)

        bodies = [p1, p2, p3]
        
        println("Testing: ω = $ω, e = $e")
        for i=1:ic.nbody
            pre_elems = ic.elements[i,:]
            elems = get_orbital_elements(s, ic)[i]
            post_elems = [elems.m, elems.P, elems.t0, elems.e * cos(elems.ω), elems.e * sin(elems.ω), rem2pi(elems.I, RoundNearest), rem2pi(elems.Ω, RoundNearest)]
            @test isapprox(pre_elems, post_elems; rtol=1e-8)
        end
    end
end

@testset "Orbit Initialization" begin
    # For n = 4 planets
    vec = [0.1, 0.2, 0.3, 0.4,
    0., 1.3, 5.0,
    0.2, 0.3, π,
    1e-4, 1e-4,
    365.242]

    optparams = OptimParameters(4, vec)
    orbparams = OrbitParameters([1e-4, 1e-4, 1e-4, 1e-4], [0.5, 0.5], 2.000, 8*365.242, [1., 1., 5., 3., 2.])

    orbit = Orbit(4, optparams, orbparams)

    anoms = get_anomalies(orbit.s, orbit.ic)

    for i=2:orbit.nplanet
        @test isapprox(rem2pi(anoms[i][2], RoundNearest), rem2pi(vec[3+i], RoundNearest); rtol=1e-8)
    end
end
