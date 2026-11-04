@isdefined(PDMPSamplers) || include(joinpath(@__DIR__, "testsetup.jl"))

# Run the Julia example of the README, so it stays in sync with the package.
@testset "README example" begin
    readme = Base.read(joinpath(@__DIR__, "..", "README.md"), String)
    example = Base.split(readme, "## Example")[2]
    code = Base.match(r"```julia\n(.*?)```"s, example).captures[1]
    code = Base.replace(code, "0.0, 10_000.0)" => "0.0, 10_000.0; progress=false)")

    sandbox = Module(:ReadmeExample)
    result = Base.include_string(sandbox, code, "README.md")
    @test result ≈ zeros(2) atol=0.2
end
