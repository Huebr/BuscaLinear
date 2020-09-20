#Julia 1.4 Environment
using Pkg
pkg"add ForwardDiff; precompile;"
pkg"add Plots; precompile;"
pkg"add PyPlot; precompile;"
import ForwardDiff  # calcula derivadas usando automatic differentiation in forward mode
using LinearAlgebra #adiciona operações de algebra linear


pyplot()#usa PyPlot como backend engine
∇ = (h,x)->ForwardDiff.gradient(h,x) #gradiente de h(x)
Hessian = (h,x)->ForwardDiff.hessian(h,x) # hessian de h(x)
Df = (h,x)->ForwardDiff.derivative(h,x);  # primeira derivada de h'(x)
D_2f = (h,x)->ForwardDiff.derivative(z->ForwardDiff.derivative(h,z),x) # segunda derivada h"(x)

#função de busca linear
function linear_search(f::Function,x::Vector,d::Vector,method::Function)
    ϕ = t->f(x+t*d)
    α = method(f,x,d)
    return (ϕ(α),x+α*d)
end

function secao_aurea(f::Function,x::Vector,d::Vector;φ::Function = t->f(x+t*d),ρ = 1/2,ϵ = 1e-9)::Float64
    θ_1 = (3 - sqrt(5))/2.0
    θ_2 = (sqrt(5) - 1)/2.0
    #Obtenção intervalo
    a=0
    b=2*ρ
    s = b / 2
    while(φ(b)<φ(s))
        a = s
        s = b
        b *= 2
    end
    #Obtenção de t
    u = a + θ_1*(b - a)
    v = a + θ_2*(b - a)
    while(b - a > ϵ)
        if(φ(u)<φ(v))
            b = v
            v = u
            u = a + θ_1*(b - a)
        else
            a = u
            u = v
            v = a + θ_2*(b - a)
        end
    end
    return (u+v)/2
end

function secao_aurea(f::Function,x::Vector,d::Vector,(a,b)::Tuple{Float64,Float64};φ::Function = t->f(x+t*d),ϵ = 1e-9)::Float64
    θ_1 = (3 - sqrt(5))/2.0
    θ_2 = (sqrt(5) - 1)/2.0
    #Obtenção de t
    u = a + θ_1*(b - a)
    v = a + θ_2*(b - a)
    while(b - a > ϵ)
        if(φ(u)< φ(v))
            b = v
            v = u
            u = a + θ_1*(b - a)
        else
            a = u
            u = v
            v = a + θ_2*(b - a)
        end
    end
    return (u+v)/2
end

function newton(f::Function,x::Vector,d::Vector;φ::Function = y->f(x+y*d),t::Float64=0.25,ϵ = 1e-9,n_iter=10000)::Float64
   iter = 0
   while (abs(Df(φ,t)) > ϵ && iter < n_iter)
         t = t - (Df(φ,t)/D_2f(φ,t))
         iter = iter + 1
   end 
   return t
end


function armijo(f::Function,x::Vector,d::Vector;φ::Function = y->f(x+y*d),t::Float64= 1.0,η::Float64 = 0.25)
    # f(x + td) > f(x) +t(∇f(x)⋅ d)
    while (φ(t)>φ(0)+ η*t*(∇(f,x)⋅d))
      t *=0.8
    end
    return t
end
