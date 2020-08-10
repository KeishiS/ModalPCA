using LinearAlgebra, Statistics
include("GeometricMedian.jl")

function HRPCA(
    inpX::Array{Float64,2},
    maxd::Int64;
    isCentered = false,
    scale = false,
    α=0.5
)::Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2}}
    (N, d) = size(inpX);
    
    gm = zeros(d);
    if !isCentered
        gm = GM(inpX);
    end
    X = inpX .- gm';

    T̂ = min(N - maxd, round(Int, N * α));
    W = zeros(d, maxd);
    opt = 0;
    rnds = rand(T̂ + 1);

    for s in 0:T̂
        Σ̂ = (X[1:N - s,:]' * X[1:N - s,:]) ./ (N - s);
        eigret = Σ̂ |> Symmetric |> A->eigen(A, d - maxd + 1:d);
        inds = sortperm(eigret.values, rev = true)[1:maxd];
        optₙ = (X[1:N - s,:] * eigret.vectors[:,inds]).^2 |> (A->mean(A, dims = 1)) |> sum;
        if optₙ > opt
            W .= eigret.vectors[:,inds];
            opt = optₙ;
        end

        αs = (X[1:N - s,:] * eigret.vectors[:,inds]).^2 |> (A->sum(A, dims = 2)) |> (v->v ./ sum(v));
        ind = map(i->rnds[s + 1] <= sum(αs[1:i]), 1:N - s) |> findfirst;

        (X[ind,:], X[N - s,:]) = (X[N - s,:], X[ind,:]);
    end

    scores = X*W
    return (gm, W, scores);
end