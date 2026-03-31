using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs

cancer_monitor_and_treatment = QuickPOMDP(
    states = [:HEALTHY, :INSITU, :INVASIVE, :DEATH],
    actions = [:WAIT, :TEST, :TREAT],
    observations = [:POS, :NEG],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if s == :HEALTHY
            return SparseCat([s, :INSITU], [0.98, 0.02])
        end
        if s == :INSITU
            if a == :TREAT
                return SparseCat([s, :HEALTHY], [0.4, 0.6])
            else
                return SparseCat([s, :INVASIVE], [0.9, 0.1])
            end
        end
        
        if s == :INVASIVE
            if a == :TREAT
                return SparseCat([s, :HEALTHY, :DEATH], [0.6, 0.2, 0.2])
            else
                return SparseCat([s, :DEATH], [0.4, 0.6])
            end
        end

        if s == :DEATH
            return Deterministic(:DEATH)
        end

    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :TEST
            if sp == :HEALTHY
                return SparseCat([:POS, :NEG], [0.05, 0.95])
            elseif sp == :INSITU
                return SparseCat([:POS, :NEG], [0.8, 0.2])
            elseif sp == :INVASIVE
                return Deterministic(:POS)
            else # death
                return Deterministic(:NEG)
            end
        elseif a == :TREAT
            if sp == :INSITU || sp == :INVASIVE
                return Deterministic(:POS)
            else
                return Deterministic(:NEG)
            end
        else
            return Deterministic(:NEG)
        end
    end,

    reward = function (s, a)
        if s == :DEATH
            return 0.0
        end
        
        if a == :WAIT
            return 1.0
        elseif a == :TEST
            return 0.8
        else # a = :TREAT
            return 0.1
        end
    end,

    initialstate = Deterministic(:HEALTHY),

    discount = 0.99
)

# evaluate with a random policy
policy = FunctionPolicy(o->:WAIT)
sim = RolloutSimulator(max_steps=100)
@show @time mean(POMDPs.simulate(sim, cancer_monitor_and_treatment, policy) for _ in 1:10_000)

