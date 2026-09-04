"""
    run!(ws::Workspace, plan::Plan, fields::Tuple, backend) -> ws

Execute every computed node of `plan` in order into the workspace; `fields` are the raw input arrays in
binding order. Allocation-free once the steps are compiled.
"""
function run!(ws::Workspace, p::Plan, fields::Tuple, backend::CB.AbstractExecutionBackend)
    _run_steps(ws.steps, fields, backend)
    return ws
end

# The step tuple is unrolled so every kernel call dispatches on a concrete step type.
@inline _run_steps(::Tuple{}, fields, backend) = nothing
@inline function _run_steps(steps::Tuple, fields, backend)
    execute!(first(steps), fields, backend)
    return _run_steps(Base.tail(steps), fields, backend)
end
run!(ws::Workspace, p::Plan, fields::NamedTuple, backend::CB.AbstractExecutionBackend) = run!(ws, p, values(fields), backend)

execute!(s::BaseStep, fields::Tuple, backend) = (boxfold!(s.out, fields, s.window, s.shape, backend); nothing)
execute!(s::CoarsenStep, fields::Tuple, backend) = (boxfold!(s.out, s.parent, s.grid, s.shape, backend); nothing)
execute!(s::ComposeStep, fields::Tuple, backend) = (compose!(s.out, s.a, s.b, s.axis, s.amap, s.bmap, backend); nothing)
execute!(s::ScanStep, fields::Tuple, backend) = (scan!(s.out, s.parent, s.axis, s.size, s.partial, s.scratch, backend); nothing)
