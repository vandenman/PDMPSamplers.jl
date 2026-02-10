"""
Detect closure boxes in PDMPSamplers.jl

This script checks for methods in PDMPSamplers that allocate Core.Box,
which can indicate performance issues from captured variables in closures.

Based on Julia PR #60478: https://github.com/JuliaLang/julia/pull/60478

Implementation compatible with Julia 1.10+
"""

using PDMPSamplers

function is_box_call(@nospecialize expr)
    if !(expr isa Expr)
        return false
    end
    if expr.head === :call || expr.head === :new
        callee = expr.args[1]
        return callee === Core.Box || (callee isa GlobalRef && callee.mod === Core && callee.name === :Box)
    end
    return false
end

function slot_name(ci, slot)::Symbol
    if slot isa Core.SlotNumber
        idx = Int(slot.id)
        if 1 <= idx <= length(ci.slotnames)
            return ci.slotnames[idx]
        end
    end
    return :unknown
end

function is_in_mods(mod::Module, recursive::Bool, mods)
    if mod in mods
        return true
    end
    if recursive
        parent = parentmodule(mod)
        parent === mod && return false
        return is_in_mods(parent, recursive, mods)
    end
    return false
end

function detect_closure_boxes(mods::Module...)
    boxes = Dict{Method, Vector{Symbol}}()
    mods = Module[mods...]
    isempty(mods) && return Pair{Method, Vector{Symbol}}[]

    function matches_module(mod::Module)
        return is_in_mods(mod, true, mods)
    end

    function scan_method!(m::Method)
        matches_module(parentmodule(m)) || return
        ci = try
            Base.uncompressed_ast(m)
        catch
            return
        end
        for stmt in ci.code
            if stmt isa Expr && stmt.head === :(=)
                lhs = stmt.args[1]
                rhs = stmt.args[2]
                if is_box_call(rhs)
                    push!(get!(Vector{Symbol}, boxes, m), slot_name(ci, lhs))
                end
            elseif is_box_call(stmt)
                push!(get!(Vector{Symbol}, boxes, m), :unknown)
            end
        end
    end

    Base.visit(Core.methodtable) do m
        scan_method!(m)
    end

    result = collect(boxes)
    sort!(result, by = entry -> (entry.first.file, entry.first.line, entry.first.name))
    return result
end

detect_closure_boxes_all_modules() = detect_closure_boxes(Base.loaded_modules_array()...)

println("=" ^ 80)
println("Detecting closure boxes in PDMPSamplers.jl")
println("=" ^ 80)
println()

# Detect boxes in the main PDMPSamplers module
println("Checking PDMPSamplers module...")
boxes = detect_closure_boxes(PDMPSamplers)

if isempty(boxes)
    println("✓ No closure boxes detected in PDMPSamplers!")
else
    println("⚠ Found $(length(boxes)) method(s) with closure boxes:")
    println()

    for (method, vars) in boxes
        println("  Method: $(method.name)")
        println("    File: $(method.file):$(method.line)")
        println("    Boxed variables: $(vars)")
        println()
    end

    println()
    println("Closure boxes can indicate performance issues. Consider:")
    println("  1. Making the captured variable a parameter instead")
    println("  2. Using let blocks to avoid captures")
    println("  3. Making captured variables const if possible")
    println("  4. Using Ref{T} explicitly if mutation is needed")
end

println()
println("=" ^ 80)
println("Checking all loaded modules (may include dependencies)...")
println("=" ^ 80)
println()

all_boxes = detect_closure_boxes_all_modules()
pdmp_boxes = filter(all_boxes) do (method, vars)
    startswith(string(method.file), "PDMPSamplers")
end

if isempty(pdmp_boxes)
    println("✓ No closure boxes detected in PDMPSamplers files across all modules!")
else
    println("⚠ Found $(length(pdmp_boxes)) PDMPSamplers method(s) with closure boxes:")
    println()

    for (method, vars) in pdmp_boxes
        println("  $(method.name) in $(method.file):$(method.line)")
        println("    Variables: $(vars)")
    end
end

println()
println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println("Total methods with boxes in PDMPSamplers: $(length(boxes))")
println("Total methods in PDMPSamplers files (all modules): $(length(pdmp_boxes))")
println()

if !isempty(boxes) || !isempty(pdmp_boxes)
    println("Run with @code_lowered to inspect specific methods:")
    println("  @code_lowered method_name(args...)")
    println()
end
