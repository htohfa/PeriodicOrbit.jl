using LinearAlgebra
using NbodyGradient
using NbodyGradient: hierarchy, amatrix
using NbodyGradient: Point

mag(x) = sqrt(dot(x,x))
mag(x,v) = sqrt(dot(x,v))

Rdotmag(R::T,V::T,h::T) where T <: AbstractFloat = V^2 - (h/R)^2 

function point2vector(x::NbodyGradient.Point)
    return [x.x, x.y, x.z]
end

using LinearAlgebra

function convert_to_elements(x::Vector{T}, v::Vector{T}, Gm::T, t::T) where T <: Real
    R = norm(x)
    v2 = dot(v, v)
    hvec = cross(x, v)
    h = norm(hvec)

    # (2.134)
    a = 1. / (2/R - v2/Gm)

    # (2.135)
    e = sqrt(1 - h^2 / (Gm * a))

    # (2.136)
    I = acos(clamp(hvec[3] / h, -1, 1))

    # (2.137) -check sign convention with eric
    if abs(I) >= 1e-8
        sinΩ = hvec[1] / (h * sin(I))
        cosΩ = -hvec[2] / (h * sin(I))
        Ω = atan(sinΩ, cosΩ)
    else
        Ω = 0.0
    end

    # (2.138)
    if abs(I) >= 1e-8
        sin_ω_plus_f = x[3] / (R * sin(I))
        cos_ω_plus_f = (1 / cos(Ω)) * (x[1] / R + sin(Ω) * sin_ω_plus_f * cos(I))
        ω_plus_f = atan(sin_ω_plus_f, cos_ω_plus_f)
    else
        ω_plus_f = atan(x[2], x[1])
    end

    # ω_plus_f = atan(x[2], x[1])

    # (2.139) assuming Rdot = (x.v)/R which is projection of velocity in radial direction
    sinf = a * (1 - e^2) * dot(x, v) / (h * e * R)
    cosf = (1 / e) * (a * (1 - e^2) / R - 1)
    f = atan(sinf, cosf)

    ω = ω_plus_f - f

    # (2.140)
    # Step 1: Solve for E explicitly from Eq. (2.42)
    cosE = (a - R) / (a * e)
    E = acos(clamp(cosE, -1, 1))

    # Adjust E based on radial velocity
    Rdot = dot(x, v) / R
    if Rdot < 0
        E = 2π - E
    end

    #  Mean anomaly M from Eq. (2.51)
    M = E - e * sin(E)

    #  Mean motion n from Eq. (2.26)
    n_mean = sqrt(Gm / a^3)

    #  τ from M and n (Eq. 2.51 rearranged)
    τ = t - M / n_mean


    return (
        a = a,
        e = e,
        I = I,
        Ω = mod2pi(Ω),
        ω = mod2pi(ω),
        f = mod2pi(f),
        M = mod2pi(M),
        E = mod2pi(E),
        τ = τ,
        n = n_mean,
        h = h
    )
end


function normvec(x::Vector{T}, v::Vector{T}, Gm::T) where T <: AbstractFloat
    # TODO: Need more comprehensive testing
    r = mag(x)
    hvec = cross(x, v)
    evec = (cross(v, hvec) / Gm) .- (x / r)

    # Unnormalized normal vector
    nvec = cross(evec, x)

    return nvec
end

function get_relative_masses(ic::InitialConditions)
    N = length(ic.m)
    M = zeros(N-1)
    G = 39.4845/(365.242 * 365.242) # AU^3 Msol^-1 Day^-2
    Hmat = hierarchy([N, ones(Int64,N-1)...])
    for i in 1:N-1
        for j in 1:N
            M[i] += abs(Hmat[i,j])*ic.m[j]
            # M[i] += abs(ic.ϵ[i,j])*ic.m[j]
        end
    end

    return G .* M
end

function get_relative_masses(masses::Vector{T}) where T <: Real
    masses = vcat(1., masses)
    N = length(masses)
    M = zeros(N-1)
    G = 39.4845/(365.242 * 365.242) # AU^3 Msol^-1 Day^-2
    Hmat = hierarchy([N, ones(Int64,N-1)...])
    for i in 1:N-1
        for j in 1:N
            M[i] += abs(Hmat[i,j])*masses[j]
            # M[i] += abs(ic.ϵ[i,j])*ic.m[j]
        end
    end

    return G .* M
end

""" Calculate the relative positions from the A-Matrix. """
function get_relative_positions(x,v,ic::InitialConditions)
    n = ic.nbody
    Hmat = hierarchy([n, ones(Int64,n-1)...])
    amat = amatrix(Hmat, ic.m)
    
    X = zeros(3,n)
    V = zeros(3,n)

    X .= permutedims(amat*x')
    V .= permutedims(amat*v')

    return Point(X), Point(V)
end


function get_orbital_elements(s::State{T}, ic::InitialConditions{T}) where T <: Real
    elems = Elements{T}[]
    μs = get_relative_masses(ic)
    X, V = get_relative_positions(s.x, s.v, ic)
    X = point2vector.(X)
    V = point2vector.(V)
    n = ic.nbody
    Hmat = hierarchy([n, ones(Int64,n-1)...])

    push!(elems, Elements(m=ic.m[1]))  # Central body
    i = 1; b = 0
    while i < ic.nbody
        if first(Hmat[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[i+b], V[i+b], μs[i+b], s.t[1])
        push!(elems, Elements(ic.m[i+1], 2π / n, 0.0, e*cos(ω), e*sin(ω), I, Ω, a, e, ω, τ))
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        i += 1
    end
    return elems
end

function get_anomalies(s::State{T}, ic::InitialConditions{T}) where T <: Real
    anoms = Vector{T}[]
    μs = get_relative_masses(ic)
    X, V = get_relative_positions(s.x, s.v, ic)
    X = point2vector.(X)
    V = point2vector.(V)

    n = ic.nbody
    Hmat = hierarchy([n, ones(Int64,n-1)...])

    i = 1; b = 0
    while i < ic.nbody
        if first(Hmat[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[i+b], V[i+b], μs[i+b], s.t[1])
        push!(anoms, [f, M, E])
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        i += 1
    end
    return anoms
end

