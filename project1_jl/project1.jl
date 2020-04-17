#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#
# import Pkg;
# Pkg.add("Plots")
#import Pkg
#Pkg.add("Plots")

#Pkg.add("PyPlot")
#Pkg.add("GR")

#didn't end up adding yet
#Pkg.add("UnicodePlots")
#Pkg.add("PlotlyJS")

# using Plots
# gr()

# Example:
using LinearAlgebra

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")


"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, x0, n, prob)
#     Basing first attempt on Nesterov's momentum gradient descent as described in Alg. 5.4 in the course textbook
 	v = vec(zeros(length(x0),1)); #set momentum to zero
	i = 1;   
#     x_history = x0; #x is 
    #x version 2
    x_history = Array{Array{Float64,1}}(undef, 1);
 	x_history[1] = x0; 
    if prob == "simple1" 
        alpha = 0.4; #learning rate
        beta = 0.3;#momentum decay, 0 = pure gradient method
    else
        alpha = 0.9; #learning rate
        beta = 0.8;#momentum decay, 0 = pure gradient method  
    end
    #version 1
#     while count(f, g) < n
#         xi_vec = vec(x_history[:,i]);
# #         if i == 1
# #             xi_vec = x_history;
# #         else
# #             xi_vec = vec(pop(x_history));
# #         end
        
#         tmp_vec = xi_vec + beta*v;
#         v = beta*v - (alpha^i)*g(tmp_vec); #adjusted Nestov's alg. to incorporate decreasing learning rate
#         normThresh = 1;
#         if norm(v) > normThresh
#             v = v / norm(v)
#         end
#         xpi_vec = xi_vec + v;
#         x_history = hcat(x_history, xpi_vec);
# #         x_history = push!(x_history, xpi_vec);
# #         println(typeof(x_history))
#         i = i + 1;
#     end
#         #end loop
# #     println(x_history)
# #     println(v)
#     return vec(x_history[:,i-1])
# #     return pop(x_history)
    
    #version 2
    while count(f, g) < n
        xi_vec = vec(x_history[i]);
#         if i == 1
#             xi_vec = x_history;
#         else
#             xi_vec = pop(x_history); #x_history should be an array of arrays, non-muatating pop
#         end
        
        tmp_vec = xi_vec + beta*v;
        v = beta*v - (alpha^i)*g(tmp_vec); #adjusted Nestov's alg. to incorporate decreasing learning rate
        normThresh = 1;
        if norm(v) > normThresh
            v = v / norm(v)
        end
        xpi_vec = xi_vec + v;
#         x_history = hcat(x_history, xpi_vec);
         x_history = push!(x_history, xpi_vec);
#         println(typeof(x_history))
        i = i + 1;
    end
#     println(collect(x_history))
        #end loop
#     println(x_history)
#     println(v)
#     return vec(x_history[:,i-1])
#     plot!, [0 0 0], [0 10 50], st = :contour)
    return vec(x_history[i-1])
end

#function to return full history of optimization path
function optimize_hist(f, g, x0, n, prob)
#     Basing first attempt on Nesterov's momentum gradient descent as described in Alg. 5.4 in the course textbook
 	v = vec(zeros(length(x0),1)); #set momentum to zero
	i = 1;   
#     x_history = x0; #x is 
    #x version 2
    x_history = Array{Array{Float64,1}}(undef, 1);
 	x_history[1] = x0; 
    if prob == "simple1" 
        alpha = 0.4; #learning rate
        beta = 0.3;#momentum decay, 0 = pure gradient method
    else
        alpha = 0.9; #learning rate
        beta = 0.8;#momentum decay, 0 = pure gradient method  
    end
    #version 1
#     while count(f, g) < n
#         xi_vec = vec(x_history[:,i]);
# #         if i == 1
# #             xi_vec = x_history;
# #         else
# #             xi_vec = vec(pop(x_history));
# #         end
        
#         tmp_vec = xi_vec + beta*v;
#         v = beta*v - (alpha^i)*g(tmp_vec); #adjusted Nestov's alg. to incorporate decreasing learning rate
#         normThresh = 1;
#         if norm(v) > normThresh
#             v = v / norm(v)
#         end
#         xpi_vec = xi_vec + v;
#         x_history = hcat(x_history, xpi_vec);
# #         x_history = push!(x_history, xpi_vec);
# #         println(typeof(x_history))
#         i = i + 1;
#     end
#         #end loop
# #     println(x_history)
# #     println(v)
#     return vec(x_history[:,i-1])
# #     return pop(x_history)
    
    #version 2
    while count(f, g) < n
        xi_vec = vec(x_history[i]);
#         if i == 1
#             xi_vec = x_history;
#         else
#             xi_vec = pop(x_history); #x_history should be an array of arrays, non-muatating pop
#         end
        
        tmp_vec = xi_vec + beta*v;
        v = beta*v - (alpha^i)*g(tmp_vec); #adjusted Nestov's alg. to incorporate decreasing learning rate
        normThresh = 1;
        if norm(v) > normThresh
            v = v / norm(v)
        end
        xpi_vec = xi_vec + v;
#         x_history = hcat(x_history, xpi_vec);
         x_history = push!(x_history, xpi_vec);
#         println(typeof(x_history))
        i = i + 1;
#           println("test")
    end
  
#     println(collect(x_history))
        #end loop
#     println(x_history)
#     println(v)
#     return vec(x_history[:,i-1])
#     plot!, [0 0 0], [0 10 50], st = :contour)
    return x_history
end
