using LinearAlgebra, BlockDiagonals, JuMP, Clarabel, Distributions, LaTeXStrings, Plots

# this function is used to calculate the stability of the periodic MJLS system
# only used at the beginning; not required after that
function dynamicsdesign()
    
    # A₁(k)   =   [   -0.5    1.95
    #                 -0.5    0.1*(sin(0.2*π*k))];
    # A₂(k)   =   [   0.5*(cos(0.2*π*k))    0.1
    #                 0.8                     0.4];

    A₁(k)   =   [   -0.5    2
                    -0.4    0.8*(sin(0.2*π*k))];
    # A₂(k)   =   [   0.5*(cos(0.2*π*k))         0.4
    #                 0.5                    0.8];

    A₂(k)   =   [   0.5*(cos(0.2*π*k))    0.5
                    0.8                   0.5];
    nₛ = 2; n =2;
    ℙ       =   [   0.8     0.2
                    0.9     0.1];


    nₛ = 2; n =2;
    # ℙ       =   [   0.6     0.4
    #                 0.5     0.5];
                    # ℙ       =   [   0.9     0.1
                    # 0.95     0.05];
    Φ       =   Matrix(1.0I,8,8);
    ϕ₁      =   Matrix(1.0I,2,2);
    ϕ₂      =   Matrix(1.0I,2,2);

    for t in 1:10
        A   =   [A₁(t-1),A₂(t-1)];
        𝒜   =   kron(ℙ,Matrix(1.0I,n^2,n^2))*BlockDiagonal([kron(A[i]', A[i]) for i in 1:nₛ]);
        # 𝒜   =   BlockDiagonal([kron(A[i]', A[i]') for i in 1:nₛ])*kron(ℙ,Matrix(1.0I,n^2,n^2));
        Φ   =   𝒜*Φ;
        ϕ₁  =   A₁(t)*ϕ₁;
        ϕ₂  =   A₂(t)*ϕ₂;
    end

    return maximum(abs.(eigvals(Φ))),maximum(abs.(eigvals(ϕ₁))),maximum(abs.(eigvals(ϕ₂)))
end

# function to check stability characteristics of the controller
function controller_MSS_test(K)
    Aall,Ball,ℙ=systemdynamics();
    nₛ=2; n=2;
    Φ       =   Matrix(1.0I,8,8);
    ϕ₁      =   Matrix(1.0I,2,2);
    ϕ₂      =   Matrix(1.0I,2,2);

    for t in 1:10
        A   =   [ Aall[t][i] + Ball[t][i]*K[i][t] for i in 1:nₛ];
        𝒜   =   kron(ℙ,Matrix(1.0I,n^2,n^2))*BlockDiagonal([kron(A[i], A[i]) for i in 1:nₛ]);
        # 𝒜   =   BlockDiagonal([kron(A[i]', A[i]') for i in 1:nₛ])*kron(ℙ,Matrix(1.0I,n^2,n^2));
        Φ   =   𝒜*Φ;
        ϕ₁  =   Aall[t][1]*ϕ₁;
        ϕ₂  =   Aall[t][2]*ϕ₂;
    end

    return maximum(abs.(eigvals(Φ))),maximum(abs.(eigvals(ϕ₁))),maximum(abs.(eigvals(ϕ₂)))
end

# function that returns the current system dynamics matrices for both modes
function systemdynamics()
    A₁(k)   =   [   -0.5    2
                    -0.4    0.8*(sin(0.2*π*k))];
    # A₂(k)   =   [   0.5*(cos(0.2*π*k))    0.1
    #                 0.8                     0.3];
                    # A₂(k)   =   [   (cos(0.2*π*k))         0.7
                    # 0.7                    0.5]

    A₂(k)   =   [   0.5*(cos(0.2*π*k))    0.5
                    0.8                   0.5];
    nₛ = 2; n =2;
    ℙ       =   [   0.8     0.2
                    0.9     0.1];

    TP      =   10;
    B₁      =   [ 1;1];
    # B₂      =   [1 ;0.5];
    B₂      =   [0;0];

    # Aall    =   [[A₁(k-1),A₂(k-1)] for k in 1:TP];
    Aall    =   [[A₁(k-1),A₂(k-1)] for k in 1:TP];
    Ball    =   [[B₁,B₂]    for k   in  1:TP];
    # Bvv = [[Bv[i],Bv[i]*0] for i in 1:length(Av)];


    return Aall,Ball,ℙ
end

function relaxed_controller(Aall,Ball,ℙ,disopt,ub,stateupperbound)
    
    T = length(Aall);
    model = Model(Clarabel.Optimizer);

    H = [1/stateupperbound    0;  0   1/stateupperbound];

    S1all = [@variable(model, [1:2, 1:2], PSD) for k = 1:T];
    S2all = [@variable(model, [1:2, 1:2], PSD) for k = 1:T];

    Y1all = [@variable(model, [1:1, 1:2]) for k = 1:T];
    Y2all = [@variable(model, [1:1, 1:2]) for k = 1:T];

    @variable(model, aux)


    # @variable(model, ρ)

    for k in 1:T
        Aₖ  =   Aall[k];
        Bₖ  =   Ball[k];

        S, Y = [S1all[k], S2all[k]], [Y1all[k], Y2all[k]];
        if k == T
            Sn = [S1all[1],S2all[1]];
        else
            Sn   = [S1all[k+1],S2all[k+1]];
        end

        for i in 1:2
            ASBY = Aₖ[i] * S[i] + Bₖ[i] * Y[i];
            𝓁    = reduce(hcat,  [sqrt(ℙ[i, j]) * I(2) for j in 1:2])
            Sₑ = [Sn[1] zeros(2, 2); zeros(2, 2) Sn[2]];
            if k == 4
                LMI2 = [          
                        0.9*S[i]               ASBY'*𝓁    
                        𝓁'*ASBY             Sₑ       
                    ]
            else
                LMI2 = [          
                        S[i]               ASBY'*𝓁    
                        𝓁'*ASBY             Sₑ       
                    ]
            end
            # LMI2 = [          
            #     S[i]        ASBY'*𝓁     Y[i]'
            #     𝓁'*ASBY     Sₑ          zeros(4,1)
            #     Y[i]        zeros(1,4)  β*I(1)
            # ]
            # LMI1_ic1 = [1 ic[1]'; ic[1] S[i]];
            # LMI1_ic2 = [1 ic[2]'; ic[2] S[i]];
            # LMI1_ic3 = [1 ic[3]'; ic[3] S[i]];
            # LMI1_ic4 = [1 ic[4]'; ic[4] S[i]];

            # if k==1
            #     @constraints(model,begin
            #     [j=1:4], Symmetric([1 ic[j]'; ic[j] S[i]]) >= 0, PSDCone()
            #     end)
            # end            

            LMI4_ub =   [   ub[i]^2                Y[i] 
                            Y[i]'               S[i] ];

            LMI5_sb =   I - H*S[i]*H';

            @constraints(model, begin
            Symmetric(LMI2) >= 0, PSDCone()
            [j=1:2], Symmetric([S[i] ASBY'; ASBY Sn[j]]) >= 0, PSDCone()
            Symmetric(LMI4_ub) >= 0, PSDCone()
            Symmetric(LMI5_sb) >= 0, PSDCone()
            end);

        end
    end
    set_optimizer_attribute(model, "verbose", disopt)
    set_optimizer_attribute(model, "tol_gap_abs", 1e-12)
    set_optimizer_attribute(model, "tol_feas", 1e-12)
    set_optimizer_attribute(model, "tol_gap_rel", 1e-12)
    @constraint(model, aux >= -tr(S1all[1]))
    @constraint(model, aux >= -tr(S2all[1]))
    @objective(model, Min, aux)
    optimize!(model)

    Ksol1 =[];
    Ksol2 =[];

    shist=[];
    for i in 1:T
        s1=JuMP.value.(S1all[i]);
        # if i==1 || i==T
        #     println("**********");
        #     println(s1);
        # end
        y1=JuMP.value.(Y1all[i]);
        push!(Ksol1,y1*inv(s1));
        s2=JuMP.value.(S2all[i]);
        y2=JuMP.value.(Y2all[i]);
        push!(Ksol2,y2*inv(s2));
        push!(shist,[s1,s2]);
    end

    # return model
    return Ksol1,Ksol2,JuMP.value.(S1all[1]),JuMP.value.(S1all[2]),shist

end
# Kr= relaxed_controller(Aall,Ball,ℙ,bic,true,[2000 2000],400)

function montecarlo_relaxed(N,disopt,T,ub,sb)
    # input: N is number of samples for monte carlo simulations
    # bic is bound of the convex hull of initial conditions
    # disopt is zero if the optimizer shouldn't publish report
    # T is the time horizon of the control simulation
    # control is 0 for uncontrolled case and 1 for controlled case
    
    bic = 250;
    x1listall=[];
    x2listall=[];
    xnlistall=[];
    u1listall=[];
    u2listall=[];
    unlistall=[];
    Clistall = [];
    cov1listall=[];
    cov2listall=[];
    Aall,Ball,ℙ=systemdynamics();
    K1,K2,S1,S2,shist = relaxed_controller(Aall,Ball,ℙ,disopt,[ub ub],sb);
    K = [K1,K2];S=[S1,S2];
    ud = Uniform(0,1);
    Vlistall=[];
    for sample in 1:N
        mode=   rand(1:2);
        # rad= rand(0:sb);
        # theta=rand(0:2*π);
        # x=[rad*cos(theta),rad*sin(theta)];
        x  =   [rand(-bic:bic),rand(-bic:bic)];
        # print(x);
        while x'*inv(shist[1][1])*x>1 || x'*inv(shist[1][2])*x>1
            # print(x'*inv(S[1])*x<1 && x'*inv(S[2])*x<1);
            x =   [rand(-bic:bic),rand(-bic:bic)];
        end
        # println(x'*inv(S[1])*x>1);
        # println(x'*inv(S[2])*x>1);
        if x'*inv(shist[1][1])*x>1 || x'*inv(shist[1][2])*x>1
            println("error");
        end
        x1list=[];x2list=[];xnlist=[];
        u1list=[];u2list=[];unlist=[];  CT=0;
        cov1list=[];cov2list=[];Vlist=[];
        push!(x1list,x[1]);
        push!(x2list,x[2]);
        push!(xnlist,norm(x));
        push!(cov1list,x[1]*x[1]');
        push!(cov2list,x[2]*x[2]');
        for t in 1:T
            time=mod(t,10);
            if time==0
                time=10;
            end
            # if t==27
            #     x=x.+50;
            # end
            st = shist[time][mode];
            push!(Vlist,x'*inv(st)*x);
            if x'*inv(st)*x>1
                println("new error");
                println(t);
            end
            if mode==1
                push!(u1list,norm(K[mode][time]*x));
            else
                push!(u2list,norm(K[mode][time]*x));
            end
            push!(unlist,norm(K[mode][time]*x));
            CT+=x'*(K[mode][time])'*K[mode][time]*x;
            x   =   (Aall[time][mode]+Ball[time][mode]*K[mode][time])*x;
            push!(x1list,x[1]);
            push!(x2list,x[2]);
            push!(xnlist,norm(x));
            push!(cov1list,x[1]*x[1]');
            push!(cov2list,x[2]*x[2]');
            mm = rand(ud);
            if mode==1
                if mm<=ℙ[1,1]
                    mode=1;
                else
                    mode=2;
                end
            else
                if mm<=ℙ[2,1]
                    mode=1;
                else
                    mode=2;
                end
            end
        end
        push!(x1listall,x1list);
        push!(x2listall,x2list);
        push!(xnlistall,xnlist);
        push!(u1listall,u1list);
        push!(u2listall,u2list);
        push!(unlistall,unlist);
        push!(cov1listall,cov1list);
        push!(cov2listall,cov2list);
        push!(Clistall,CT);
        push!(Vlistall,Vlist);
    end
    #propagating theoretic covariance
    cov0=[  (2*bic)^2/12    0
            0               (2*bic)^2/12    ];
    cov1=cov0*0.5;cov2=cov0*0.5;
    covlist=[copy(cov1)+copy(cov2)];
    for k in 1:T
        time=mod(k,10);
        if time==0
            time=10;
        end
        cov =   [copy(cov1),copy(cov2)];
        ϕ1   =   Aall[time][1]+Ball[time][1]*K[1][time];
        ϕ2   =   Aall[time][2]+Ball[time][2]*K[2][time];
        ϕ   =[ϕ1,ϕ2];
        cov1 =   sum([ℙ[i,1]*ϕ[i]*cov[i]*ϕ[i]'  for i in 1:2]);
        cov2 =   sum([ℙ[i,2]*ϕ[i]*cov[i]*ϕ[i]'  for i in 1:2]);
        push!(covlist,cov1+cov2);
    end
    c1=[3*sqrt(covlist[i][1,1]) for i in 1:T+1];
    c2=[3*sqrt(covlist[i][2,2]) for i in 1:T+1];
    return x1listall,x2listall,u1listall,u2listall,Clistall,cov1listall,cov2listall,c1,c2,xnlistall,unlistall,Vlistall
end

function plots(uc)
    xl=montecarlo_relaxed(1000,true,30,[125 125],250)
    p1=plot(xl[1],label=nothing,xlab=L"t",ylab=L"x_1");p2=plot(xl[2],label=nothing,xlab=L"t",ylab=L"x_2");
    plot!(p1,xl[8],lw=2, linestyle=:dash,color=:black,label=L"3\sigma"); plot!(p1,-xl[8],lw=2, linestyle=:dash,color=:black,label=nothing);
    hline!(p1,[250],label=L"x_{b}",color=:grey,linestyle=:dash,lw=2);hline!(p1,[-250],label=nothing,color=:grey,linestyle=:dash,lw=2)
    plot!(p2,xl[9],lw=2, linestyle=:dash,color=:black,label=L"3\sigma"); plot!(p2,-xl[9],lw=2, linestyle=:dash,color=:black,label=nothing);
    hline!(p2,[250],label=L"x_{b}",color=:grey,linestyle=:dash,lw=2);hline!(p2,[-250],label=nothing,color=:grey,linestyle=:dash,lw=2)
    pall=plot(p1,p2,layout = @layout [a;b])
    cv1=mean(reduce(hcat,xl[6]),dims=2);
    cv2=mean(reduce(hcat,xl[7]),dims=2);
    pc1 = plot(cv1,label=nothing,xlab=L"t",ylab=L"\mathbb{E}[x_1(t)x_1^\top(t)]");pc2=plot(cv2,label=nothing,xlab=L"t",ylab=L"\mathbb{E}[x_2(t)x_2^\top(t)]");
    pcall=plot(pc1,pc2,layout = @layout [a;b])
    un = plot(xl[11],label=nothing,xlab=L"t",ylab=L"\||u\||",xtickfontsize=18,ytickfontsize=18,labelfontsize=22,legendfontsize=16);
    hline!([125],label=L"u_{\max}",color=:grey,linestyle=:dash,lw=2)
    xn = plot(xl[10],label=nothing,xlab=L"t",ylab=L"\||x\||",xtickfontsize=18,ytickfontsize=18,labelfontsize=22,legendfontsize=16);
    hline!([250],label=L"\|x\|_b",color=:grey,linestyle=:dash,lw=2)
    return pall,xn,un
end

# Kr= relaxed_controller(Aall,Ball,ℙ,true,[125 125],250)