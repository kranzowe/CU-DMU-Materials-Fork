#!/bin/bash
#SBATCH --job-name=hw2_vi
#SBATCH --output=hw2_%j.out
#SBATCH --error=hw2_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nice=100

# ============================================
# Memory limit - SLURM will kill job if exceeded
# This protects the node from your job eating all RAM
# Adjust --mem above based on what you need
# ============================================

# Also set a soft limit via ulimit (in KB)
# 30GB soft limit - gives warning before SLURM hard kill
# ulimit -v 31457280

echo "==================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Memory limit: $SLURM_MEM_PER_NODE MB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "==================================="

# Check current node memory before starting
echo ""
echo "Node memory status before job:"
free -h
echo ""

# ============================================
# Run Julia with memory-conscious settings
# ============================================
julia --threads=$SLURM_CPUS_PER_TASK \
      --heap-size-hint=24G \
      -e '
println("Julia started with $(Threads.nthreads()) threads")
println("Starting memory: $(round(Sys.free_memory() / 1e9, digits=2)) GB free")

# Memory monitoring function
function check_memory()
    free_gb = Sys.free_memory() / 1e9
    total_gb = Sys.total_memory() / 1e9
    used_gb = total_gb - free_gb
    println("Memory: $(round(used_gb, digits=2))/$(round(total_gb, digits=2)) GB used")
    
    # Warn if getting low
    if free_gb < 4.0
        @warn "Low memory! Only $(round(free_gb, digits=2)) GB free"
    end
    
    return free_gb
end

# Run with try-catch to handle OOM gracefully
try
    check_memory()
    
    println("\n=== Including hw2_solutions.jl ===\n")
    include("hw2_solutions.jl")
    
    println("\n=== Finished ===")
    check_memory()
    
catch e
    if isa(e, OutOfMemoryError)
        println("\n!!! OUT OF MEMORY !!!")
        println("Job ran out of memory. Try:")
        println("  1. Using sparse matrices")
        println("  2. Reducing problem size")
        println("  3. Requesting more memory with --mem")
    else
        println("\n!!! ERROR !!!")
        showerror(stdout, e)
        println()
        showerror(stdout, e, catch_backtrace())
    end
    exit(1)
end
'

echo ""
echo "==================================="
echo "Job finished: $(date)"
echo "Node memory status after job:"
free -h
echo "==================================="