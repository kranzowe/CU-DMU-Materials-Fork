using DMUStudent.HW2
using POMDPs: states, actions, discount, stateindex, convert_s, transition
using POMDPTools: ordered_states, render, weighted_iterator
import Cairo, Fontconfig # Needed in some cases for rendering the value function on grid world
using Debugger
using SparseArrays
using CUDA
##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW2 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 3
############

# @show actions(grid_world) # prints the actions. In this case each action is a Symbol. Use ?Symbol to find out more.

# T = transition_matrices(grid_world)
# display(T) # this is a Dict that contains a transition matrix for each action

# @show T[:left][1, 2] # the probability of transitioning between states with indices 1 and 2 when taking action :left



function value_iteration(m)
    γ = discount(m)
    A = actions(m)
    T_sparse = transition_matrices(m, sparse = true)
    T_sparse_transposed = Dict(action => sparse(T_sparse[action]') for action in A) # for nzrange gotta flip which is confusin
    #@show T_sparse
    R = reward_vectors(m)
    states_defined = states(m)
    num_states = length(states_defined)
    V_old = zeros(num_states)
    V_new = zeros(num_states)
    count = 0
    
    while true
        for i in 1:num_states

            max_Q = -Inf
            for action in A

                Q_next_states = 0.0 # gonna sum the next states which should reduce compute alot since its a sparse matrix now

                for idx in nzrange(T_sparse_transposed[action], i) # non zero indexs of the sparse T for this action for this col i which is the current state
                    j = rowvals(T_sparse_transposed[action])[idx] # this is actually the index of the state can go to
                    prob = nonzeros(T_sparse_transposed[action])[idx] # extract the prob of transition from the spare struct
                    Q_next_states += prob * V_old[j]
                end
                
                max_Q =  max(max_Q, R[action][i] + (γ * Q_next_states)) # only assign if its new max
            
            end
            V_new[i] = max_Q
            
        end
        count += 1
        println(count)

        # inf norm
        if maximum(abs.(V_new .- V_old)) < 1e-6
            
            break
        end

        # so instead of copying each time, we just swap the variable
        V_new, V_old = V_old, V_new # ac

    end

    return V_new
end

function matrix_value_iteration(m)
    γ = discount(m)
    A = actions(m)
    T_sparse = transition_matrices(m, sparse = true)
    
    R = reward_vectors(m)
    num_states = length(states(m))
    
    V_old = zeros(num_states)
    V_new = zeros(num_states)
    count = 0
    
    while true
        
        Q_mat = hcat([R[a] .+ γ .* (T_sparse[a] * V_old) for a in A]...)

        V_new .= vec(maximum(Q_mat, dims=2))
        
        count += 1
        println(count)

        # inf norm
        if maximum(abs.(V_new .- V_old)) < 1e-6
            break
        end

        # so instead of copying each time, we just swap the variable
        V_new, V_old = V_old, V_new # ac

    end

    return V_new
end

function no_storage_value_iteration(m)
    γ = discount(m)
    A = actions(m)

    R = reward_vectors(m)
    S = ordered_states(m)
    num_states = length(S)
    V_old = zeros(num_states)
    V_new = zeros(num_states)
    count = 0
    
    while true
        for (i, s) in enumerate(S)

            max_Q = -Inf
            for action in A

                Q = R[action][i]

                # gotta add the next states
                d = transition(m, s, action)
                for (sp, p) in weighted_iterator(d)
                    j = stateindex(m, sp) # this is actually the index of the state can go to
                    Q += p * V_old[j]
                end
                
                max_Q =  max(max_Q, Q) # only assign if its new max
            
            end
            V_new[i] = max_Q
            
        end
        count += 1
        println(count)

        # inf norm
        if maximum(abs.(V_new .- V_old)) < 1e-6
            
            break
        end

        # so instead of copying each time, we just swap the variable
        V_new, V_old = V_old, V_new # ac

    end

    return V_new
end

# function gpu_value_iteration(map)
#     γ = Float32(discount(m))
#     A = collect(actions(m))
#     num_actions = Int32(length(A))

#     #needed dims to run this on GPU
#     nhbins = nlabels(m.hbins)
#     nhdotbins = nlabels(m.hdotbins)
#     ndbins = nlabels(m.dbins)
#     num_states = nhbins * nhdotbins * nhbins * ndbins

#     h_centers = CuArray(Float32.(collect(bincenters(m.hbins))))
#     hdot_centers = CuArray(Float32.(collect(bincenters(m.hdotbins))))
#     d_centers = CuArray(Float32.(collect(bincenters(m.dbins))))

#     actions_gpu = CuArray(Float32.(A))

#     R = reward_vectors(m)
#     R_matrix = hcat([Float32.(R_cpu[a]) for a in A]...)
#     R_gpu = CuArray(R_matrix)

    
#     num_states = length(S)
#     V_old = CUDA.zeros(Float32, num_states)
#     V_new = CUDA.zeros(num_states)

#     count = 0
#     threads_per_block = 256
#     blocks = cld(num_states, threads_per_block)

#     while true
#         @cuda threads = threads_per_block blocks=blocks kernel_value_iteration!(

#         )



#     end
# end

# function kernel_value_iteration!(V_new, V_old, R, γ, actions, num_actions,
#                                 h_centers, hdot_centers, d_centers,
#                                 )

#@enter(value_iteration(grid_world))
# m = grid_world
# V = value_iteration(m) # replace this with value_iteration(m)
# # If you are in an environment with multimedia capability (e.g. VSCode, Jupyter, Pluto), use this:
# #display(render(grid_world, color=V)) # In the REPL, this will output an annoying amount of text
# # If you are in the REPL or want to save a png, use this:
# using Compose: draw, PNG
# draw(PNG("value.png"), render(m, color=V))

############
# Question 4
############



# You can create an mdp object representing the problem with the following:
m = UnresponsiveACASMDP(15)
# @show actions(m)

# s = first(states(m))
# @show transition(m, s, 0)
# transition_matrices and reward_vectors work the same as for grid_world, however this problem is much larger, so you will have to exploit the structure of the problem. In particular, you may find the docstring of transition_matrices helpful:
# display(@doc(transition_matrices))
#@enter(value_iteration(m))
V = no_storage_value_iteration(m)

# @show HW2.evaluate(V)
HW2.evaluate(V, "owen.kranz@colorado.edu")

########
# Extras
########

# The comments below are not needed for the homework, but may be helpful for interpreting the problems or getting a high score on the leaderboard.

# Both UnresponsiveACASMDP and grid_world implement the POMDPs.jl interface. You can find complete documentation here: https://juliapomdp.github.io/POMDPs.jl/stable/api/#Model-Functions

# To convert from physical states to indices in the transition function, use the stateindex function
# IMPORTANT NOTE: YOU ONLY NEED TO USE STATE INDICES FOR THIS ASSIGNMENT, using the states may help you make faster specialized code for the ACAS problem, but it is not required
# using POMDPs: states, stateindex

# s = first(states(m))
# @show si = stateindex(m, s)

# # # To convert from a state index to a physical state in the ACAS MDP, use convert_s:
# using POMDPs: convert_s

# @show s = convert_s(ACASState, si, m)

# # # To visualize a state in the ACAS MDP, use
# render(m, (s=s,))
