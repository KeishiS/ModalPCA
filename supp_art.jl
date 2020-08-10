using LinearAlgebra, Statistics, StatsBase, Random, DelimitedFiles, PGFPlotsX;

include("CoP.jl");
include("DPCP.jl");
include("GeometricMedian.jl");
include("HRPCA.jl");
include("R1PCA.jl");
include("REAPER.jl");
include("RPCAOM.jl");
include("TMPCA.jl");
include("ModalPCA.jl");

PCAmethods = [
  "CoP"
  "R1"
  "REAPER"
  "DPCP"
  "HRPCA"
  "TMPCA"
  "RPCAOM"
  "ModalPCA"
];
dists = ["normal", "laplace"];


function SpecDist(A::Array{Float64,2}, B::Array{Float64,2})::Float64
  svdret = A*inv(A'*A)*A' - B*inv(B'*B)*B' |> svd;
  return min(1.0, svdret.S |> maximum) |> asin;
end

rad2deg(x::Float64) = 180 * x / π

function test_modalpca(;
  N::Int64 = 300, d::Int64 = 10, Loop::Int64 = 2,
  seedval::Int64 = 123, contrib::Float64 = 0.95,
  by::Float64 = 0.05, dist::String = "normal"
)
  ϵrng = range(0,0.45, step=by)
  ttl = sum([1.0/i^3 for i in 1:d])
  K = findfirst(e->e/ttl >= contrib, [1.0/i^3 for i in 1:d] |> cumsum)

  specDists = zeros(length(PCAmethods), length(ϵrng));
  specDistsStd = zeros(length(PCAmethods), length(ϵrng));

  for j in 1:length(ϵrng)
    ϵ = ϵrng[j]
    rets = zeros(length(PCAmethods), Loop);
    for l in 1:Loop
      Random.seed!(j*l*seedval);
      x = zeros(N, d);
      if dist == "normal"
        x[1:round(Int, N*(1-ϵ)), :] .= rand(MvNormal(zeros(d), diagm([1.0/i^3 for i in 1:d])), round(Int, N*(1-ϵ)))' |> Matrix;
      elseif dist == "laplace"
        for i in 1:d
          x[1:round(Int, N*(1-ϵ)), i] .= rand(Laplace(1.0, 10/i^2), round(Int, N*(1-ϵ)));
        end
      end
      if round(Int, N*ϵ) > 0
        x[round(Int, N*(1-ϵ))+1:N, :] .= rand(Uniform(-1,1.5), round(Int, N*ϵ), d);
      end

      resCoP = CoP(x)[2][:,1:K]
      rets[findfirst(isequal("CoP"), PCAmethods), l] = SpecDist(resCoP, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      resR1 = R1PCA(x, K)[2][:,1:K]
      rets[findfirst(isequal("R1"), PCAmethods), l] = SpecDist(resR1, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      resREAPER = REAPER(x, K)[2][:,1:K]
      rets[findfirst(isequal("REAPER"), PCAmethods), l] = SpecDist(resREAPER, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      resDPCP = DPCP(x)[2][:,1:K]
      rets[findfirst(isequal("DPCP"), PCAmethods), l] = SpecDist(resDPCP, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      resHR = HRPCA(x, K)[2][:,1:K]
      rets[findfirst(isequal("HRPCA"), PCAmethods), l] = SpecDist(resHR, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      restm = TMPCA(x, maxd=K, jₛ=10, p=3)[2][:,1:K]
      rets[findfirst(isequal("TMPCA"), PCAmethods), l] = SpecDist(restm, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      resrpcaom = RPCAOM(x, maxd=K)[2][:,1:K]
      rets[findfirst(isequal("RPCAOM"), PCAmethods), l] = SpecDist(resrpcaom, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;

      resModalPCA = ModalPCA(x)[2][:,1:K]
      rets[findfirst(isequal("ModalPCA"), PCAmethods), l] = SpecDist(resModalPCA, Matrix(1.0I,d,d)[:,1:K]) |> rad2deg;
      
      println("[DONE] ϵ: $(ϵ), loop: $(l)");
    end
    for i in 1:length(PCAmethods)
      specDists[i,j] = median(rets[i,:])
      specDistsStd[i,j] = std(rets[i,:])
    end
  end

  Dict{Symbol,Any}(
    :specDists => specDists,
    :specDistsStd => specDistsStd,
    :ϵrng => ϵrng
  )
end


for dt in dists
  N = 200;
  d = 20;

  local ret = test_modalpca(N=N, d=d, Loop=100, seedval = 111, dist = dt);
  out_path = "$(dt)_eps_ratio.pdf";
  
  ax = @pgf Axis(
    {
      legend_style = {
        legend_pos = "outer north east"
      },
      grid,
    },
    [
      PlotInc(
        Table(x=ret[:ϵrng], y=ret[:specDists][i,:])
      )
      for i in 1:length(PCAmethods)
    ]...,
    [
      LegendEntry(method)
      for method in PCAmethods
    ]...
  );
  pgfsave(out_path, ax);
end