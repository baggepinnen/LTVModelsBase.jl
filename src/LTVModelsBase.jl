module LTVModelsBase

using Parameters
# Interface exports
export Trajectory, AbstractModel, AbstractCost, ModelAndCost,f,
dc,calculate_cost,calculate_final_cost, predict, simulate, df,costfun, LTVStateSpaceModel,
SimpleLTVModel, covariance

export rms, sse, nrmse


rms(x)      = sqrt(mean(x.^2))
sse(x)      = x⋅x
nrmse(y,yh) = 100 * (1-rms(y-yh)./rms(y-mean(y)))


# Trajectory =========================================
@with_kw mutable struct Trajectory
    x::Matrix{Float64}
    u::Matrix{Float64}
    y::Matrix{Float64}
    xu::Matrix{Float64}
    nx::Int
    nu::Int
    function Trajectory(x,u)
        @assert size(x,2) == size(u,2) "The second dimension of x and u (time) must be the same"
        x,u,y = x[:,1:end-1],u[:,1:end-1],y[:,2:end]
        new(x,u,y,[x;u],size(x,1), size(u,1))
    end
    function Trajectory(x,u,y)
        @assert size(x,2) == size(u,2) == size(y,2) "The second dimension of x,u and y (time) must be the same"
        new(x,u,y,[x;u],size(x,1), size(u,1))
    end
end
Base.length(t::Trajectory) = size(t.x,2)

Base.start(t::Trajectory) = 1
Base.next(t::Trajectory, state) = ((t.x[:,state], t.u[:,state]), state+1)
Base.done(t::Trajectory, state) = state == length(t)


# Model interface ====================================
"""
Model interface, implement the following functions\n
see also `AbstractCost`, `ModelAndCost`
```
fit_model(::Type{AbstractModel}, batch::Batch)::AbstractModel

predict(model::AbstractModel, x, u)

function df(model::AbstractModel, x, u, I::UnitRange)
    return fx,fu,fxx,fxu,fuu
end
```
"""
abstract type AbstractModel end

abstract type LTVModel <: AbstractModel end

abstract type LTVStateSpaceModel <: LTVModel end

mutable struct SimpleLTVModel{T} <: LTVStateSpaceModel
    At::Array{T,3}
    Bt::Array{T,3}
    extended::Bool
end
function SimpleLTVModel(At::Array{T,3},Bt::Array{T,3},extend::Bool) where T
    if extend
        At = cat(3,At,At[:,:,end])
        Bt = cat(3,Bt,Bt[:,:,end])
    end
    return SimpleLTVModel{T}(At,Bt,extend)
end

# SimpleLTVModel(At,Bt,extend::Bool) = SimpleLTVModel{eltype(At)}(At,Bt,extend)


"""
    model = fit_model(::Type{AbstractModel}, x,u)::AbstractModel

Fits a model to data
"""
function fit_model(::Type{AbstractModel}, x,u)::AbstractModel
    error("This function is not implemented for your type")
    return model
end

"""
    fit_model!(model::AbstractModel, x,u)::AbstractModel

Refits a model to new data
"""
function fit_model!(model::AbstractModel, x,u)::AbstractModel
    error("This function is not implemented for your type")
    return model
end

"""
    xnew = predict(model::AbstractModel, t::Trajectory [, i])
    xnew = predict(model::AbstractModel, x, u [, i])

Predict the next state given the current state and action
"""
function predict(model::AbstractModel, x, u, i)
    error("This function is not implemented for your type")
    return xnew
end

predict(model::AbstractModel, t::Trajectory, args...) = predict(model, t.x, t.u, args...)

"""
    xnew = simulate(model::AbstractModel, t::Trajectory)
    xnew = simulate(model::AbstractModel, x0, u)

Simulate system forward in time given the initial state and actions
"""
function simulate(model::AbstractModel, x0, u)
    error("This function is not implemented for your type")
    return xnew
end

simulate(model::AbstractModel, t::Trajectory) = simulate(model, t.x[:,1], t.u)


"""
    fx,fu,fxx,fxu,fuu = df(model::AbstractModel, x, u)

Get the linearized dynamics at `x`,`u`
"""
function df(model::AbstractModel, x, u)
    error("This function is not implemented for your type")
    return fx,fu,fxx,fxu,fuu
end

function covariance(model::AbstractModel, x, u)
    cov(x[:,2:end]-predict(model, x, u)[:,1:end-1], 2)
end
# Model interface ====================================


# Cost interface ====================================
"""
Cost interface, implement the following functions\n
see also `AbstractModel`, `ModelAndCost`
```
function calculate_cost(::Type{AbstractCost}, x::AbstractVector, u)::Number

function calculate_cost(::Type{AbstractCost}, x::AbstractMatrix, u)::AbstractVector

function calculate_final_cost(::Type{AbstractCost}, x::AbstractVector)::Number

function dc(::Type{AbstractCost}, x, u)
    return cx,cu,cxx,cuu,cxu
end
```
"""
abstract type AbstractCost end

function calculate_cost(c::AbstractCost, x::AbstractVector, u)::Number
    error("This function is not implemented for your type")
    return c
end

function calculate_cost(c::AbstractCost, x::AbstractMatrix, u)::AbstractVector
    error("This function is not implemented for your type")
    return c
end

function calculate_final_cost(c::AbstractCost, x::AbstractVector)::Number
    error("This function is not implemented for your type")
    return c
end

function dc(c::AbstractCost, x, u)
    error("This function is not implemented for your type")
    return cx,cu,cxx,cuu,cxu
end
# Cost interface ====================================


"""
1. Define types that implement the interfaces `AbstractModel` and `AbstractCost`.
2. Create object modelcost = ModelAndCost(model, cost)
3. Run macro @define_modelcost_functions(modelcost). This macro defines the following functions
```
f(x, u, i)  = f(modelcost, x, u, i)
fT(x)       = fT(modelcost, x)
df(x, u, I) = df(modelcost, x, u, I)
```
see also `AbstractModel`, `AbstractCost`
"""
type ModelAndCost
    model::AbstractModel
    cost::AbstractCost
end

function f(modelcost::ModelAndCost, x, u, i)
    predict(modelcost.model, x, u, i)
end

function costfun(modelcost::ModelAndCost, x, u)
    calculate_cost(modelcost.cost, x, u)
end

"""
    fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(modelcost::ModelAndCost, x, u)

Get the linearized dynamics and cost at `x`,`u`
"""
function df(modelcost::ModelAndCost, x, u)
    fx,fu,fxx,fxu,fuu = df(modelcost.model, x, u)
    cx,cu,cxx,cuu,cxu = dc(modelcost.cost, x, u)
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end




end # module
