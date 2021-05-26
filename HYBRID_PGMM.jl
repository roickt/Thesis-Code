using DataFrames, CSV, Tables, LinearAlgebra, Statistics, StatsBase, Random, Clustering, NaNMath, MPI, Base.Threads

BLAS.set_num_threads(1)

function read_job_cmd()
    gmin = parse(Float64, ARGS[1])
    gmax = parse(Float64, ARGS[2])
    qmin = parse(Float64, ARGS[3])
    qmax = parse(Float64, ARGS[4])
    tmin = parse(Float64, ARGS[5])
    tmax = parse(Float64, ARGS[6])
    n_in = parse(Float64, ARGS[7])
    p_in = parse(Float64, ARGS[8])
    data_in = ARGS[9]
    return gmin, gmax, qmin, qmax, tmin, tmax, n_in, p_in, data_in
end

function work(x_df, gg, qq, mm, n, p)

    function convergtest_new(l, at, v_max, v, n, it, g, tol)
        append!(l, Float64(0))
        append!(at, 0)
        flag = 0
        for i in 1:n
            summ = 0
            for j in 1:g
                summ += exp(v[i, j] - v_max[i])
            end
            l[it] += log(summ) + v_max[i]
            if isnan(l[it]) || isinf(l[it])
                return -1, l, at
            end
        end
        if it > 1
            if l[it] < l[it-1]
                return -1, l, at
            end
        end
        if it > 3
            at[it-1] = (l[it] - l[it - 1]) / (l[it - 1] - l[it - 2])
            if at[it - 1] < 1.0
                l_inf = l[it - 1] + (l[it] - l[it - 1]) / (1 - at[it - 1])
                if abs(l_inf - l[it]) < tol
                    flag = 1
                end
            end
        end
        return flag, l, at
    end

    function update_mu(n1, x, z, g, n, p)
        mu = zeros(g, p)
        ksum = 0
        for j in 1:g
            for k in 1:p
                mu[j, k] = sum(z[:, j] .* x[:, k])
            end
            mu[j, :] /= n1[j]
            ksum += sum(mu[j, :])
        end
        return mu
    end

    function woodbury(x, lambda, psi, mu, p, q, lt)
        xm = transpose(x-mu)
        lhs = sum(xm .^ 2) ./ psi
        lvec = xm ./ psi
        tvec = lvec * lambda
        temp = lt ./ psi
        result = temp * lambda + Matrix{Float64}(I, q, q)
        cp = inv(result)
        result = cp * lt
        cvec = tvec * result
        rhs = sum(cvec .* xm) ./ psi
        output = lhs .- rhs
        return output
    end

    function woodbury2(x, lambda, psi, mu, p, q, lt)
        psi = transpose(psi)
        xm = transpose(x-mu)
        lhs = sum((xm .^ 2) ./ psi)
        lvec = xm ./ psi
        tvec = lvec * lambda
        temp = lt ./ psi
        result = temp * lambda + Matrix{Float64}(I, q, q)
        cp = inv(result)
        result = cp * lt
        cvec = tvec * result
        rhs = sum((cvec .* xm) ./ psi)
        output = lhs .- rhs
        return output
    end

    function update_z(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        lt = transpose(lambda)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                a = woodbury(x0 , lambda, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z2(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        lt = transpose(lambda)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                a = woodbury2(x0 , lambda, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z3(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        lt = transpose(lambda)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                a = woodbury(x0 , lambda, psi[j], mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z4(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        lt = transpose(lambda)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                psi0 = psi[(j-1)*p+1:(j-1)*p+p]
                a = woodbury2(x0 , lambda, psi0, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z5(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                lambda0 = transpose(reshape(lambda[:, j], q, p))
                lt = transpose(lambda0)
                a = woodbury(x0 , lambda0, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z6(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                lambda0 = transpose(reshape(lambda[:, j], q, p))
                lt = transpose(lambda0)
                a = woodbury2(x0 , lambda0, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z7(x::Array{Float64}, z::Array{Float64}, lambda::Array{Float64}, psi::Array{Float64}, mu::Array{Float64}, pyi::Array{Float64}, log_c, n::Int64, g::Int64, p::Int64, q::Int64)
        x0 = zeros(p)::Array{Float64}
        mu0 = zeros(p)::Array{Float64}
        v0 = zeros(g)::Array{Float64}
        v = zeros(n, g)::Array{Float64}
        max_v = zeros(n)::Array{Float64}
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                lambda0 = transpose(reshape(lambda[:, j], q, p))
                lt = transpose(lambda0)
                a = woodbury(x0 , lambda0, psi[j], mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z8(x, z, lambda, psi, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        for i in 1:n
            for j in 1:g
                x0 = x[i, :]
                mu0 = mu[j, :]
                psi0 = psi[j, :]
                lambda0 = transpose(reshape(lambda[:, j], q, p))
                lt = transpose(lambda0)
                a = woodbury2(x0 , lambda0, psi0, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function known_z(class, z, n, g)
        z = zeros(n, g)
        for i in 1:n
            if class[i] != 0
                for j in 1:g
                    z[i, j] = 0
                    if j == class[i]
                        z[i, j] = 1
                    end
                end
            end
        end
        return z
    end

    function update_stilde(x, z, mu, g, n, p, pyi)
        sampcovtilde = zeros(p, p)
        for t in 1:g
            dsub = x .- reshape(mu[t, :],1,:)
            wv = weights(z[:,t])
            sampcovtilde += pyi[t]*(scattermat(dsub, wv; mean=0)/sum(z[:,t]))
        end
        return sampcovtilde
    end

    function update_sg(x::Array{Float64}, z::Array{Float64}, mu::Array{Float64}, g::Int64, n::Int64, p::Int64, n1::Array{Float64})
        sampcovtilde = Dict()
        for t in 1:g
            dsub = x .- reshape(mu[t, :],1,:)
            wv = weights(z[:,t])
            sampcovtilde[t] = (scattermat(dsub, wv; mean=0)/sum(z[:,t]))
        end
        return sampcovtilde
    end

    function update_beta1(psi, llambda, p, q)
        lhs = transpose(llambda) / psi
        cp = lhs * llambda
        result = cp .+ Matrix{Float64}(I, q, q)
        rhs = inv(result)
        res = cp * rhs
        rhs = res * lhs
        beta = lhs - rhs
        return beta
    end

    function update_beta2(psi, llambda, p, q)
        lhs = transpose(llambda ./ psi)
        cp = lhs * llambda
        result = cp .+ Matrix{Float64}(I, q, q)
        rhs = inv(result)
        res = cp * rhs
        rhs = res * lhs
        beta = lhs - rhs
        return beta
    end

    function update_theta(beta, llambda, sampcovtilde, p, q)
        r_1 = beta * llambda
        r_2 = beta * sampcovtilde
        r_3 = r_2 * transpose(beta)
        theta = Matrix{Float64}(I, q, q) .- r_1 .+ r_3
        return theta
    end

    function update_lambda(beta, s, theta, p, q)
        res1 = s * transpose(beta)
        llambda = res1 * inv(theta)
        return llambda
    end

    function update_lambda2(beta, s, theta, n1, psi, p, q, g)
        res2 = zeros(p, q)
        for j in 1:g
            tran = transpose(beta[j])
            res1 = s[j] * tran
            if j == 1
                for i in 1:p
                    for k in 1:q
                        res2[i, k] = res1[i, k] * n1[j] / psi[j]
                    end
                end
            else
                for i in 1:p
                    for k in 1:q
                        res2[i, k] += res1[i, k] * n1[j] / psi[j]
                    end
                end
            end
        end
        res3 = zeros(q, q)
        for j in 1:g
            if j == 1
                for i in 1:q
                    for k in 1:q
                        res3[i, k] = theta[j][i, k] * n1[j] / psi[j]
                    end
                end
            else
                for i in 1:q
                    for k in 1:q
                        res3[i, k] += theta[j][i, k] * n1[j] / psi[j]
                    end
                end
            end
        end
        llambda = res2 * inv(res3)
        return llambda
    end

    function update_lambda_cuu(beta, s, theta, n1, psi, p, q, g)
        llambda = zeros(p, q)
        res2 = zeros(p, q)
        for j in 1:g
            tran = transpose(beta[j])
            res1 = s[j] * tran
            if j == 1
                for i in 1:p
                    for k in 1:q
                        res2[i, k] = res1[i, k] * n1[j] / psi[(j-1)*p+i]
                    end
                end
            else
                for i in 1:p
                    for k in 1:q
                        res2[i, k] += res1[i, k] * n1[j] / psi[(j-1)*p+i]
                    end
                end
            end
        end
        res3 = zeros(q, q)
        for ii in 1:p
            for j in 1:g
                if j == 1
                    for i in 1:q
                        for k in 1:q
                            res3[i, k] = theta[j][i, k] * n1[j] / psi[(j-1)*p+ii]
                        end
                    end
                else
                    for i in 1:q
                        for k in 1:q
                            res3[i, k] += theta[j][i, k] * n1[j] / psi[(j-1)*p+ii]
                        end
                    end
                end
            end
            llambda0 = transpose(res2[ii, :]) * inv(res3)
            llambda0 = vec(llambda0)
            for j in 1:q
                llambda[ii, j] = llambda0[j]
            end
        end
        return llambda
    end

    function update_psi(llambda, beta, sampcovtilde, p, q)
        res1 = llambda * beta
        res2 = res1 * sampcovtilde
        psi = sum(diag(sampcovtilde) .- diag(res2)) / p
        return psi
    end

    function update_psi2(llambda, beta, sampcovtilde, p, q)
        res1 = llambda * beta
        res2 = res1 * sampcovtilde
        psi = diag(sampcovtilde) .- diag(res2)
        return psi
    end

    function update_psi3(llambda, beta, sampcovtilde, theta, p, q)
        res1 = llambda * beta
        res2 = diag(res1 * sampcovtilde)
        temp = transpose(llambda)
        res1 = llambda * theta
        res3 = diag(res1 * temp)
        psi = sum(diag(sampcovtilde) .- 2*res2 .+ res3) / p
        return psi
    end

    function update_psi_cuu(llambda, beta, sampcovtilde, theta, p, q, g)
        res2 = zeros(g, p)
        for j in 1:g
            res1 = llambda * beta[j]
            result = diag(res1 * sampcovtilde[j])
            res2[j, :] = result
        end
        res3 = zeros(g, p)
        temp = transpose(llambda)
        for j in 1:g
            res1 = llambda * theta[j]
            result = diag(res1 * temp)
            res3[j, :] = result
        end
        psi = zeros(g*p)
        for j in 1:g
            for i in 1:p
                psi[(j-1)*p+i] = sampcovtilde[j][i, i] - 2*res2[j, i] + res3[j, i]
            end
        end
        return psi
    end

    function update_psi_ucc(llambda, beta, sampcovtilde, p, q, pyi, g)
        res2 = zeros(g, p)
        for j in 1:g
            llambda0 = transpose(reshape(llambda[:, j], q, p))
            res1 = llambda0 * beta[j]
            result = diag(res1 * sampcovtilde[j])
            res2[j, :] = result
        end
        psi = 0
        for j in 1:g
            for i in 1:p
                psi += pyi[j] * (sampcovtilde[j][i, i] - res2[j, i])
            end
        end
        psi = psi / p
        return psi
    end

    function update_psi_ucu(llambda, beta, sampcovtilde, p, q, pyi, g)
        res2 = zeros(g, p)
        for j in 1:g
            llambda0 = transpose(reshape(llambda[:, j], q, p))
            res1 = llambda0 * beta[j]
            result = diag(res1 * sampcovtilde[j])
            res2[j, :] = result
        end
        psi = zeros(p)
        for i in 1:p
            for j in 1:g
                psi[i] += pyi[j] * (sampcovtilde[j][i, i] - res2[j, i])
            end
        end
        return psi
    end

    function update_det_sigma_new(llambda, psi, log_detpsi, p, q)
        tmp2 = update_beta1(psi, llambda, p, q)
        tmp = tmp2 * llambda
        tmp2 = -1 .* tmp + Matrix{Float64}(I, q, q)
        det_sigma_new = log_detpsi - NaNMath.log(det(tmp2))
        return det_sigma_new
    end

    function update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
        tmp2 = update_beta2(psi, llambda, p, q)
        tmp = tmp2 * llambda
        tmp2 = -1 .* tmp + Matrix{Float64}(I, q, q)
        det_sigma_new = log_detpsi - NaNMath.log(det(tmp2))
        return det_sigma_new
    end

    function update_omega(llambda, delta, beta, sampcovtilde, theta, p, q)
        result_1 = llambda * beta
        result_2 = diag(result_1 * sampcovtilde)
        temp = transpose(llambda)
        result_1 = llambda * theta
        result_3 = diag(result_1 * temp)
        omega = sum((diag(sampcovtilde) .- 2*result_2 .+ result_3) ./ delta) / p
        return omega
    end

    function update_omega2(llambda, delta, beta, sampcovtilde, p, q)
        result_1 = llambda * beta
        result_2 = diag(result_1 * sampcovtilde)
        omega = sum((diag(sampcovtilde) .- result_2) ./ delta) / p
        return omega
    end

    function update_delta(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g)
        result_2 = zeros(g, p)
        for j in 1:g
            result_1 = llambda * beta[j]
            result = diag(result_1 * sampcovtilde[j])
            result_2[j, :] = result
        end
        result_3 = zeros(g, p)
        temp = transpose(llambda)
        for j in 1:g
            result_1 = llambda * theta[j]
            result = diag(result_1 * temp)
            result_3[j, :] = result
        end
        temp1 = zeros(p)
        for i in 1:p
            for j in 1:g
                temp1[i] += (sampcovtilde[j][i, i] - 2*result_2[j, i] +
                result_3[j, i]) * n1[j] / omega[j]
            end
        end
        lagrange = sum(NaNMath.log.(temp1)) / p
        lagrange = (exp(lagrange) - n)/2
        delta = temp1 ./ (n + 2*lagrange)
        return delta
    end

    function update_delta2(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g)
        result_2 = zeros(g, p)
        for j in 1:g
            llambda0 = transpose(reshape(llambda[:, j], q, p))
            result_1 = llambda0 * beta[j]
            result = diag(result_1 * sampcovtilde[j])
            result_2[j, :] = result
        end
        result_3 = zeros(g, p)
        for j in 1:g
            llambda0 = transpose(reshape(llambda[:, j], q, p))
            temp = transpose(llambda0)
            result_1 = llambda0 * theta[j]
            result = diag(result_1 * temp)
            result_3[j, :] = result
        end
        temp1 = zeros(p)
        lagrange = 0
        for i in 1:p
            for j in 1:g
                temp1[i] += (sampcovtilde[j][i, i] - 2*result_2[j, i] +
                result_3[j, i]) * n1[j] / omega[j]
                lagrange += NaNMath.log(temp1[i])
            end
        end
        lagrange = lagrange / p
        lagrange = (exp(lagrange) - n)/2
        delta = temp1 ./ (n + 2*lagrange)
        return delta
    end

    function update_delta3(llambda, omega, beta, sampcovtilde, theta, n1, p, q)
        result_1 = llambda * beta
        result_2 = diag(result_1 * sampcovtilde)
        temp = transpose(llambda)
        result_1 = llambda * theta
        result_3 = diag(result_1 * temp)
        temp1 = diag(sampcovtilde) .- 2*result_2 .+ result_3
        lagrange = sum(NaNMath.log.(temp1)) / p
        lagrange = exp(lagrange) / omega
        lagrange = (lagrange-1) * (n1/2)
        delta = temp1/((1+(2*lagrange/n1))*omega)
        return delta
    end

    function update_z9(x, z, lambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        lt = transpose(lambda)
        for i in 1:n
            for j in 1:g
                psi = omega[j] .* delta
                x0 = x[i, :]
                mu0 = mu[j, :]
                a = woodbury2(x0, lambda, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z10(x, z, lambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        for i in 1:n
            for j in 1:g
                psi = omega[j] .* delta
                x0 = x[i, :]
                mu0 = mu[j, :]
                lambda0 = transpose(reshape(lambda[:, j], q, p))
                lt = transpose(lambda0)
                a = woodbury2(x0 , lambda0, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z11(x, z, lambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        lt = transpose(lambda)
        for i in 1:n
            for j in 1:g
                psi = omega .* delta[j, :]
                x0 = x[i, :]
                mu0 = mu[j, :]
                a = woodbury2(x0 , lambda, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function update_z12(x, z, lambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        x0 = zeros(p)
        mu0 = zeros(p)
        v0 = zeros(g)
        v = zeros(n, g)
        max_v = zeros(n)
        for i in 1:n
            for j in 1:g
                psi = omega .* delta[j, :]
                x0 = x[i, :]
                mu0 = mu[j, :]
                lambda0 = transpose(reshape(lambda[:, j], q, p))
                lt = transpose(lambda0)
                a = woodbury2(x0 , lambda0, psi, mu0, p, q, lt)
                e = a / 2.0 * (-1.0)
                v[i, j] = e + log(pyi[j]) - log_c[j]
            end
            v0 = v[i, :]
            max_v[i] = maximum(v0)
            d_alt = 0
            vmv = v[i, :] .- max_v[i]
            d_alt += sum(exp.(vmv))
            z[i, :] = exp.(vmv) ./ d_alt
        end
        return z, v, max_v
    end

    function aecm(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
            beta = update_beta1(psi, llambda, p, q)
            theta = update_theta(beta, llambda, sampcovtilde, p, q)
            llambda = update_lambda(beta, sampcovtilde, theta, p, q)
            psi = update_psi(llambda, beta, sampcovtilde, p, q)
            log_detpsi = p * NaNMath.log(psi)
            log_detsig = update_det_sigma_new(llambda, psi, log_detpsi, p, q)
            log_c = (p / 2) * log(2 * pi) + 0.5 * log_detsig
            tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + 1
        bic = 2*l[it-1] - paras * log(n)
        return z, bic, llambda, psi
    end

    function aecm2(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
            beta = update_beta2(psi, llambda, p, q)
            theta = update_theta(beta, llambda, sampcovtilde, p, q)
            llambda = update_lambda(beta, sampcovtilde, theta, p, q)
            psi = update_psi2(llambda, beta, sampcovtilde, p, q)
            log_detpsi = 0
            for i in 1:p
                log_detpsi += NaNMath.log(psi[i])
            end
            log_detsig = update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
            log_c = (p / 2) * log(2 * pi) + 0.5 * log_detsig
            tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm3(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                beta[j] = update_beta1(psi[j], llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            llambda = update_lambda2(beta, sampcovtilde, theta, n1, psi, p, q, g)
            psi = Dict()
            for j in 1:g
                psi[j] = update_psi3(llambda, beta[j], sampcovtilde[j], theta[j],
                p, q)
            end
            log_detpsi = zeros(g)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                log_detpsi[j] = p * NaNMath.log(psi[j])
                log_detsig[j] = update_det_sigma_new(llambda, psi[j], log_detpsi[j],
                 p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + g
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm4(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi0 = psi[(j-1)*p+1 : (j-1)*p+p]
                beta[j] = update_beta2(psi0, llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, psi, p,
                q, g)
            psi = update_psi_cuu(llambda, beta, sampcovtilde, theta, p, q, g)
            log_detpsi = zeros(g)
            for j in 1:g
                log_detpsi[j] += sum(NaNMath.log.(psi[(j-1)*p+1 : (j-1)*p+p]))
            end
            log_detsig = zeros(g)
            for j in 1:g
                psi0 = psi[(j-1)*p+1 : (j-1)*p+p]
                log_detsig[j] = update_det_sigma_new2(llambda, psi0, log_detpsi[j],
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + g*p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm5(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta1(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            psi = update_psi_ucc(llambda, beta, sampcovtilde, p, q, pyi, g)
            log_detpsi = 0
            for j in 1:p
                log_detpsi += sum(NaNMath.log(psi))
            end
            log_detsig = zeros(g)
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new(llambda0, psi, log_detpsi,
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm6(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            psi = update_psi_ucu(llambda, beta, sampcovtilde, p, q, pyi, g)
            log_detpsi = 0
            for j in 1:p
                log_detpsi += sum(NaNMath.log(psi[j]))
            end
            log_detsig = zeros(g)
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi,
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm7(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta1(psi[j], llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            psi = zeros(g)
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                psi[j] = update_psi(llambda0, beta[j], sampcovtilde[j], p, q)
            end
            log_detpsi = p * NaNMath.log.(psi)
            log_detsig = zeros(g)
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new(llambda0, psi[j],
                log_detpsi[j], p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm8(z, x, cls, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        psi = transpose(reshape(psi, p, g))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi0 = psi[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi0, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                psi_temp = update_psi2(llambda0, beta[j], sampcovtilde[j], p, q)
                psi[j, :] = psi_temp
            end
            log_detpsi = zeros(g)
            for j in 1:g
                psi0 = psi[j, :]
                log_detpsi[j] = sum(NaNMath.log.(psi0))
            end
            log_detsig = zeros(g)
            for j in 1:g
                psi0 = psi[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi0, log_detpsi[j],
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g*p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function aecm9(z, x, cls, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        delta = ones(p)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega[j] .* delta
                beta[j] = update_beta2(psi, llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            llambda = update_lambda2(beta, sampcovtilde, theta, n1, omega, p, q, g)
            for j in 1:g
                omega[j] = update_omega(llambda, delta, beta[j], sampcovtilde[j],
                theta[j], p, q)
            end
            delta = update_delta(llambda, omega, beta, sampcovtilde, theta, n1,
            p, q, n, g)
            log_detpsi = zeros(g)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega[j] .* delta
                log_detpsi[j] = p * NaNMath.log(omega[j])
                log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi[j],
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n,
            g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + g + (p-1)
        bic = 2*l[it-1] - paras * log(n)
        #println(l)
        for i in 1:p
            omega[g+i] = delta[i]
        end
        return z, bic, llambda, omega
    end

    function aecm10(z, x, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        delta = ones(p)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega[j] .* delta
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                omega[j] = update_omega2(llambda0, delta, beta[j], sampcovtilde[j],
                p, q)
            end
            delta = update_delta2(llambda, omega, beta, sampcovtilde, theta, n1,
            p, q, n, g)
            log_detpsi = zeros(g)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega[j] .* delta
                log_detpsi[j] = p * NaNMath.log(omega[j])
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi[j],
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n,
            g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g + (p-1)
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        for i in 1:p
            omega[g+i] = delta[i]
        end
        return z, bic, llambda, omega
    end

    function aecm11(z, x, cls, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        delta = ones(g, p)
        omega = omega[1]
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega .* delta[j, :]
                beta[j] = update_beta2(psi, llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            del0 = vec(transpose(delta))
            llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, del0, p,
            q, g)
            omega = 0
            for j in 1:g
                delta0 = delta[j, :]
                omega += pyi[j] * update_omega(llambda, delta0, beta[j],
                sampcovtilde[j], theta[j], p, q)
            end
            for j in 1:g
                delta0 = delta[j, :]
                delta_temp = update_delta3(llambda, omega, beta[j], sampcovtilde[j],
                 theta[j], n1[j], p, q)
                delta[j, :] = delta_temp
            end
            log_detpsi = p * NaNMath.log(omega)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega .* delta[j, :]
                log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi,
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n, g,
            p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + 1 + g*(p-1)
        #print(l)
        bic = 2*l[it-1] - paras * log(n)
        psi = zeros(g * p + 1)
        psi[1] = omega
         k = 1
        for j in 1:g
            for i in 1:p
                psi[k] = delta[j, i]
                k += 1
            end
        end
        return z, bic, llambda, psi
    end

    function aecm12(z, x, cls, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        delta = ones(g, p)
        omega = omega[1]
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
                z = known_z(cls, z, n, g)
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega .* delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            omega = 0
            for j in 1:g
                delta0 = delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                omega += pyi[j] * update_omega2(llambda0, delta0, beta[j],
                sampcovtilde[j], p, q)
            end
            for j in 1:g
                delta0 = delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                delta_temp = update_delta3(llambda0, omega, beta[j],
                sampcovtilde[j], theta[j], n1[j], p, q)
                delta[j, :] = delta_temp
            end
            log_detpsi = p * NaNMath.log(omega)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega .* delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi,
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n,
            g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            z = known_z(cls, z, n, g)
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1 + g*(p-1)
        #print(l)
        bic = 2*l[it-1] - paras * log(n)
        psi = zeros(g * p + 1)
        psi[1] = omega
        k = 1 #global
        for j in 1:g
            for i in 1:p
                psi[k] = delta[j, i]
                k += 1
            end
        end
        return z, bic, llambda, psi
    end

    function claecm(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
            beta = update_beta1(psi, llambda, p, q)
            theta = update_theta(beta, llambda, sampcovtilde, p, q)
            llambda = update_lambda(beta, sampcovtilde, theta, p, q)
            psi = update_psi(llambda, beta, sampcovtilde, p, q)
            log_detpsi = p * NaNMath.log(psi)
            log_detsig = update_det_sigma_new(llambda, psi, log_detpsi, p, q)
            log_c = (p / 2) * log(2 * pi) + 0.5 * log_detsig
            tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + 1
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        #println(l)
        return z, bic, llambda, psi
    end

    function claecm2(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
            beta = update_beta2(psi, llambda, p, q)
            theta = update_theta(beta, llambda, sampcovtilde, p, q)
            llambda = update_lambda(beta, sampcovtilde, theta, p, q)
            psi = update_psi2(llambda, beta, sampcovtilde, p, q)
            log_detpsi = 0
            for i in 1:p
                log_detpsi += NaNMath.log(psi[i])
            end
            log_detsig = update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
            log_c = (p / 2) * log(2 * pi) + 0.5 * log_detsig
            tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function claecm3(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                beta[j] = update_beta1(psi[j], llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            llambda = update_lambda2(beta, sampcovtilde, theta, n1, psi, p, q, g)
            psi = zeros(g)
            for j in 1:g
                psi[j] = update_psi3(llambda, beta[j], sampcovtilde[j], theta[j],
                p, q)
            end
            log_detpsi = zeros(g)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                log_detpsi[j] = p * NaNMath.log(psi[j])
                log_detsig[j] = update_det_sigma_new(llambda, psi[j], log_detpsi[j],
                 p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + g
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        #println(l)
        return z, bic, llambda, psi
    end

    function claecm4(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi0 = psi[(j-1)*p+1 : (j-1)*p+p]
                beta[j] = update_beta2(psi0, llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, psi, p,
                q, g)
            psi = update_psi_cuu(llambda, beta, sampcovtilde, theta, p, q, g)
            log_detpsi = zeros(g)
            for j in 1:g
                log_detpsi[j] += sum(NaNMath.log.(psi[(j-1)*p+1 : (j-1)*p+p]))
            end
            log_detsig = zeros(g)
            for j in 1:g
                psi0 = psi[(j-1)*p+1 : (j-1)*p+p]
                log_detsig[j] = update_det_sigma_new2(llambda, psi0, log_detpsi[j],
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + g*p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function claecm5(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta1(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            psi = update_psi_ucc(llambda, beta, sampcovtilde, p, q, pyi, g)
            log_detpsi = 0
            for j in 1:p
                log_detpsi += sum(NaNMath.log(psi))
            end
            log_detsig = zeros(g)
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new(llambda0, psi, log_detpsi,
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function claecm6(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            psi = update_psi_ucu(llambda, beta, sampcovtilde, p, q, pyi, g)
            log_detpsi = 0
            for j in 1:p
                log_detpsi += sum(NaNMath.log(psi[j]))
            end
            log_detsig = zeros(g)
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi,
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function claecm7(z::Array{Float64}, x::Array{Float64}, q::Int64, p::Int64, g::Int64, n::Int64, llambda::Array{Float64}, psi::Array{Float64}, tol::Float64)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta1(psi[j], llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            psi = zeros(g)::Array{Float64}
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                psi[j] = update_psi(llambda0, beta[j], sampcovtilde[j], p, q)
            end
            log_detpsi = p * NaNMath.log.(psi)
            log_detsig = zeros(g)::Array{Float64}
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new(llambda0, psi[j],
                log_detpsi[j], p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g
        #println(l)
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function claecm8(z, x, q, p, g, n, llambda, psi, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        psi = transpose(reshape(psi, p, g))
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi0 = psi[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi0, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                psi_temp = update_psi2(llambda0, beta[j], sampcovtilde[j], p, q)
                psi[j, :] = psi_temp
            end
            log_detpsi = zeros(g)
            for j in 1:g
                psi0 = psi[j, :]
                log_detpsi[j] = sum(NaNMath.log.(psi0))
            end
            log_detsig = zeros(g)
            for j in 1:g
                psi0 = psi[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi0, log_detpsi[j],
                p, q)
            end
            log_c = zeros(g)
            for j in 1:g
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g*p
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        return z, bic, llambda, psi
    end

    function claecm9(z, x, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        delta = ones(p)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega[j] .* delta
                beta[j] = update_beta2(psi, llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            llambda = update_lambda2(beta, sampcovtilde, theta, n1, omega, p, q, g)
            for j in 1:g
                omega[j] = update_omega(llambda, delta, beta[j], sampcovtilde[j],
                theta[j], p, q)
            end
            delta = update_delta(llambda, omega, beta, sampcovtilde, theta, n1,
            p, q, n, g)
            log_detpsi = zeros(g)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega[j] .* delta
                log_detpsi[j] = p * NaNMath.log(omega[j])
                log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi[j],
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n,
            g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + g + (p-1)
        bic = 2*l[it-1] - paras * log(n)
        #println(l)
        for i in 1:p
            omega[g+i] = delta[i]
        end
        return z, bic, llambda, omega
    end

    function claecm10(z, x, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        delta = ones(p)
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega[j] .* delta
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                omega[j] = update_omega2(llambda0, delta, beta[j], sampcovtilde[j],
                p, q)
            end
            delta = update_delta2(llambda, omega, beta, sampcovtilde, theta, n1,
            p, q, n, g)
            log_detpsi = zeros(g)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega[j] .* delta
                log_detpsi[j] = p * NaNMath.log(omega[j])
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi[j],
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n,
            g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g + (p-1)
        bic = 2*l[it-1] - paras * log(n)
        #println(bic)
        for i in 1:p
            omega[g+i] = delta[i]
        end
        return z, bic, llambda, omega
    end

    function claecm11(z, x, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = transpose(reshape(llambda[1:q*p], q, p))
        delta = ones(g, p)
        omega = omega[1]
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega .* delta[j, :]
                beta[j] = update_beta2(psi, llambda, p, q)
            end
            theta = Dict()
            for j in 1:g
                theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
            end
            del0 = vec(transpose(delta))
            llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, del0, p,
            q, g)
            omega = 0
            for j in 1:g
                delta0 = delta[j, :]
                omega += pyi[j] * update_omega(llambda, delta0, beta[j],
                sampcovtilde[j], theta[j], p, q)
            end
            for j in 1:g
                delta0 = delta[j, :]
                delta_temp = update_delta3(llambda, omega, beta[j], sampcovtilde[j],
                 theta[j], n1[j], p, q)
                delta[j, :] = delta_temp
            end
            log_detpsi = p * NaNMath.log(omega)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega .* delta[j, :]
                log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi,
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n, g,
            p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + p*q - q*(q-1)/2 + 1 + g*(p-1)
        #print(l)
        bic = 2*l[it-1] - paras * log(n)
        psi = zeros(g * p + 1)
        psi[1] = omega
         k = 1
        for j in 1:g
            for i in 1:p
                psi[k] = delta[j, i]
                k += 1
            end
        end
        return z, bic, llambda, psi
    end

    function claecm12(z, x, q, p, g, n, llambda, omega, tol)
        l = []
        at = []
        it = 1
        flag = 0
        log_c = 0
        llambda = reshape(llambda, q*p, g)
        delta = ones(g, p)
        omega = omega[1]
        while flag == 0
            n1 = sum(z, dims=1)
            pyi = n1/n
            mu = update_mu(n1, x, z, g, n, p)
            if it > 1
                tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n,
                g, p, q)
                z = tmpzv[1]
                v = tmpzv[2]
                max_v = tmpzv[3]
            end
            sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
            beta = Dict()
            for j in 1:g
                psi = omega .* delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                beta[j] = update_beta2(psi, llambda0, p, q)
            end
            theta = Dict()
            for j in 1:g
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
            end
            for j in 1:g
                t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
                llambda[:, j] = vec(transpose(t_llambda))
            end
            omega = 0
            for j in 1:g
                delta0 = delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                omega += pyi[j] * update_omega2(llambda0, delta0, beta[j],
                sampcovtilde[j], p, q)
            end
            for j in 1:g
                delta0 = delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                delta_temp = update_delta3(llambda0, omega, beta[j],
                sampcovtilde[j], theta[j], n1[j], p, q)
                delta[j, :] = delta_temp
            end
            log_detpsi = p * NaNMath.log(omega)
            log_detsig = zeros(g)
            log_c = zeros(g)
            for j in 1:g
                psi = omega .* delta[j, :]
                llambda0 = transpose(reshape(llambda[:, j], q, p))
                log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi,
                p, q)
                log_c[j] = (p/2) * log(2*pi) + 0.5 * log_detsig[j]
            end
            tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n,
            g, p, q)
            z = tmpzv[1]
            v = tmpzv[2]
            max_v = tmpzv[3]
            stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
            flag = stop[1]
            l = stop[2]
            at = stop[3]
            it += 1
        end
        paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1 + g*(p-1)
        #print(l)
        bic = 2*l[it-1] - paras * log(n)
        psi = zeros(g * p + 1)
        psi[1] = omega
        k = 1 #global
        for j in 1:g
            for i in 1:p
                psi[k] = delta[j, i]
                k += 1
            end
        end
        return z, bic, llambda, psi
    end

    function run_pgmm(x, z, bic, cls, q, p, g, n, model, clust, lambda, psi, tol)
        functype = [aecm, aecm2, aecm3, aecm4, aecm5, aecm6, aecm7, aecm8, aecm9,aecm10, aecm11, aecm12]
        functype2 = [claecm, claecm2, claecm3, claecm4, claecm5, claecm6, claecm7, claecm8, claecm9, claecm10, claecm11, claecm12]
        if clust != 0
            func = functype[model]
            out = func(z, x, cls, q, p, g, n, lambda, psi, tol)
        else
            func2 = functype2[model]
            out = func2(z, x, q, p, g, n, lambda, psi, tol)
        end
        z = out[1]
        bic = out[2]
        lambda = out[3]
        psi = out[4]
        return z, bic, lambda, psi
    end

    function init_load(x4, z4, g4, n4, p4, q4)
        sampcov = Dict()
        for j in 1:g4
            sampcov[j] = zeros(p4, p4)
        end
        mu = zeros(g4, p4)
        n = sum(z4, dims=1)
        pyi = n / size(x4)[1]
        for j in 1:g4
            temp = transpose(x4) * z4[:, j]
            mu[j, :] = sum(temp, dims=2) / n[j]
        end
        for t in 1:g4
            dsub = x4 .- reshape(mu[t, :],1,:)
            wv = weights(z4[:,t])
            sampcov[t] = scattermat(dsub, wv; mean=0)/sum(z4[:,t])
        end
        lambda = Dict()
        lambda1_temp = zeros(1, p4*q4*g4)
        s = 1 #global
        for j in 1:g4
            evecs = eigvecs(sampcov[j])
            evecs = evecs[:, end:-1:1]
            evals = eigvals(sampcov[j])
            evals = evals[end:-1:1]
            for i in 1:p4
                for k in 1:q4
                    lambda1_temp[s] = sqrt(evals[k]) * -1*evecs[i]
                    s += 1
                end
            end
        end
        lambda["sep"] = lambda1_temp
        psi = Dict()
        lam_mat = Dict()
        k4 = 0 #global
        for j in 1:g4
            lam_mat[j] = reshape(lambda1_temp[(1+k4):(j*p4*q4)], q4, p4)
            lam_mat[j] = transpose(lam_mat[j])
            k4 = p4*q4*j
        end
        temp_p6 = zeros(p4) #global
        for j in 1:g4
            t1 = diag(sampcov[j] - (lam_mat[j] * transpose(lam_mat[j])))
            temp_p6 = temp_p6 .+ pyi[j] * abs.(t1)
        end
        psi[6] = temp_p6
        psi[5] = sum(psi[6]) / p4
        psi_tmp = zeros(g4, p4)
        for j in 1:g4
            t1 = diag(sampcov[j] - (lam_mat[j] * transpose(lam_mat[j])))
            psi_tmp[j, :] = abs.(t1)
        end
        psi[7] = mean(psi_tmp, dims=2)
        psi[8] = vec(transpose(psi_tmp))
        stilde = zeros(p4, p4) #global
        for j in 1:g4
            stilde = stilde + pyi[j] * sampcov[j]
        end
        evecs = eigvecs(stilde)
        evecs = -evecs[:, end:-1:1]
        evals = eigvals(stilde)
        evals = evals[end:-1:1]
        lambda1_tilde = deepcopy(lambda1_temp)
        s = 1 #global
        for i in 1:p4
            for j in 1:q4
                lambda1_tilde[s] = sqrt(evals[j]) * evecs[i]
                s += 1
            end
        end
        lambda["tilde"] = lambda1_tilde
        lam_mat[1] = reshape(lambda1_tilde[1:(p4*q4)], q4, p4)
        lam_mat[1] = transpose(lam_mat[1])
        psi[2] = abs.(diag(stilde - (lam_mat[1] * transpose(lam_mat[1]))))
        psi[1] = sum(psi[2]) / p4
        psi_tmp = zeros(g4, p4)
        for j in 1:g4
            t1 = diag(sampcov[j] - (lam_mat[1] * transpose(lam_mat[1])))
            psi_tmp[j, :] = abs.(t1)
        end
        psi[3] = mean(psi_tmp, dims=2)
        psi[4] = vec(transpose(psi_tmp))
        psi[9] = vcat(psi[3], zeros(p4))
        psi[10] = vcat(psi[7], zeros(p4))
        psi[11] = vcat(psi[1], zeros(g4*p4))
        psi[12] = vcat(psi[5], zeros(g4*p4))
        lambda["psi"] = psi
        return lambda
    end

    n_dim = deepcopy(n)
    tol=0.1
    icl = false
    class = zeros(n_dim)
    class_ind = 0
    z_init = zeros(n_dim, gg)
    if gg == 1
        z_ind = ones(n_dim)
    else
        Random.seed!(123456)
        z_ind = kmeans(transpose(x_df), gg, init=:kmpp).assignments
    end
    for i in 1:n_dim
        z_init[i, z_ind[i]] = 1
    end
    z_in = deepcopy(z_init)
    lmbda = init_load(x_df, z_in, gg, n, p, qq)
    if mm == 1 || mm == 2 || mm == 3 || mm == 4 || mm == 9 || mm == 1
        lam_temp = deepcopy(lmbda["tilde"])
    else
        lam_temp = deepcopy(lmbda["sep"])
    end
    psi_temp = deepcopy(lmbda["psi"][mm])
    #temp = run_pgmm(x, z_in, 0, class, q1, p, g1, n, models_num[m], class_ind, lam_temp, psi_temp, tol)
    temp = run_pgmm(x_df, z_in, 0, class, qq, p, gg, n_dim, mm, class_ind, lam_temp, psi_temp, tol)
    #println("G = ", gg, " Q = ", qq, " Model = ", mm, " BIC = ", temp[2])
    if !isnan(temp[2])
        if icl && gg > 1
            z_mat_tmp = temp[1]
            z_mat_tmp = reshape(z_mat_tmp, n, gg)
            mapz = zeros(n)
            for i9 in 1:n
                mapz[i9] = argmax(z_mat_tmp[i9, :])
            end
            icl2 = 0
            for i9 in 1:n
                icl2 = icl2 + log(z_mat_tmp[i9, mapz[i9]])
            end
            bic_out = bic_out + 2 * icl2
        end
        z_best = temp[1]
        bic_out = temp[2]
        lambda_best = temp[3]
        psi_best = temp[4]
    else
        z_best = zeros(n_dim*gg)
        bic_out = -Inf
    end
    return bic_out, z_best
end

function master(comm, size, rank)
    time_in = time()
    message_recv = zeros(4)
    message_send = zeros(7)
    bic_save = -Inf
    g_b = -Inf
    q_b = -Inf
    m_b = -Inf
    icl=false
    zstart=2
    tol=0.1
    get_params = read_job_cmd()
    rg = get_params[1]:get_params[2]
    rq = get_params[3]:get_params[4]
    rm = get_params[5]:get_params[6]
    n = get_params[7]
    p = get_params[8]
    data_name = get_params[9]
    println("PGMM paramters G = $rg Q = $rq and Model = $rm. Applying to Data = $data_name.")
    df_raw = CSV.read(data_name, delim = " ", header=0)
    df_x = convert(Matrix, df_raw)
    x = df_x[:,1:trunc(Int, p)]
    x1 = vec(x)
    #models_all = ["CCC","CCU","CUC","CUU","UCC","UCU","UUC","UUU","CCUU","UCUU","CUCU","UUCU"]
    tasks_all = length(rq)
    task_all_index = 0
    num_workers = size - 1
    closed_workers = 0
    while closed_workers < num_workers
        ping = MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, comm)
        flag = ping[1]
        if flag == 1
            source = MPI.Get_source(ping[2])
            tag = MPI.Get_tag(ping[2])
            MPI.Recv!(message_recv,source,tag,comm)
            if tag == 0
                if task_all_index < tasks_all
                    message_send = [rq[task_all_index+1], minimum(rg), maximum(rg), minimum(rm), maximum(rm), n, p]
                    MPI.Send(message_send, source, 3, comm)
                    MPI.Send(x1, source, 3, comm)
                    task_all_index += 1
                else
                    message_send = zeros(3)
                    MPI.Send(message_send, source, 2, comm)
                    closed_workers += 1
                end
            elseif tag == 1
                if message_recv[1] > bic_save && message_recv[1] != 0
                    bic_save = message_recv[1]
                    g_b = trunc(Int, message_recv[2])
                    q_b = trunc(Int, message_recv[3])
                    m_b = trunc(Int, message_recv[4])
                end
            end
        end
    end
    time_out = time()
    println("The best BIC = $bic_save for G = $g_b, Q = $q_b, and M = $m_b.")
    println("Time taken is $(time_out - time_in)")
end

function slave(comm, size, rank)
    message_recv = zeros(4)
    message_send = zeros(7)
    data = Array{Float64}(undef, 0)
    go = true
    while go == true
        MPI.Send(message_recv, 0, 0, comm)
        given_work = MPI.Recv!(message_send, 0, MPI.MPI_ANY_TAG, comm)
        tag = MPI.Get_tag(given_work)
        if tag == 3
            qq = trunc(Int, message_send[1])
            gg_min = trunc(Int, message_send[2])
            gg_max = trunc(Int, message_send[3])
            mm_min = trunc(Int, message_send[4])
            mm_max = trunc(Int, message_send[5])
            n = trunc(Int, message_send[6])
            p = trunc(Int, message_send[7])
            data = zeros(n*p)
            MPI.Recv!(data, 0, MPI.MPI_ANY_TAG, comm)
            x_df = reshape(data, n, p)
            #maxrun = length(gg_min:gg_max) * length(mm_min:mm_max)
            res_save = fill!(zeros(gg_max, mm_max), -Inf)
            runs = collect(Iterators.product(gg_min:gg_max, mm_min:mm_max))
            maxrun = length(runs)
            thread_job = zeros(length(runs),2)
            for k in 1:length(runs)
                for j in 1:2
                    thread_job[k,j] = runs[k][j]
                end
            end
            #println("Worker $rank has $(Threads.nthreads()) threads available for Q = $qq.")
            Threads.@threads for k in 1:maxrun
                gg = trunc(Int, thread_job[k,1])
                mm = trunc(Int, thread_job[k,2])
                @inbounds res_save[gg,mm] = work(x_df, gg, qq, mm, n, p)[1]
                #println("Got BIC = $(res_save[gg,mm]) for G = $gg Q = $qq and Model = $mm.")
            end
            message_recv[1] = maximum(res_save)
            message_recv[2] = argmax(res_save)[1]
            message_recv[3] = qq
            message_recv[4] = argmax(res_save)[2]
            MPI.Send(message_recv, 0, 1, comm)
        elseif tag == 2
            message_recv = zeros(4)
            MPI.Send(message_recv, 0, 1, comm)
            go = false
        end
    end
end


#cd("$(homedir())/Desktop/NO_INCLUDE")
#mpiexec -n 2 julia HYBRID_PGMM.jl 2 3 1 2 1 12 62 461 alon.txt
#mpiexec julia MPI_PGMM.jl 2 3 1 12 1 12 62 461 alon.txt

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        master(comm, size, rank)
    else
        slave(comm, size, rank)
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
