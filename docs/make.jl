using Documenter, ReduceDims

makedocs(;
    modules=[ReduceDims],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/tkf/ReduceDims.jl/blob/{commit}{path}#L{line}",
    sitename="ReduceDims.jl",
    authors="Takafumi Arakaki <aka.tkf@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/tkf/ReduceDims.jl",
)
