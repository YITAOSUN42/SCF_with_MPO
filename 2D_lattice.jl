# -*- coding: utf-8 -*-
using ITensors
using ITensorMPS
using LinearAlgebra
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using Quantics
using HDF5

ITensors.op(::OpName"sigma_plus",::SiteType"Qubit") =
 [0 1
  0 0]

ITensors.op(::OpName"sigma_minus",::SiteType"Qubit") =
 [0 0
  1 0]

#1. General tools
#####################################
#For the break when moving to next row
function break_chain(L_chain, num_site, sites)    
    xvals =  range(1, num_site; length=num_site)
    f(x) =  if isinteger(x / L_chain)
        0
    else
        1
    end 

    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
    density_mpo = outer(density_mps',density_mps) 
    
    for i in 1:Int(log2(num_site))
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return  density_mpo 
end


#all intra-row hopping
function intrachain_hopping(L_chain, hop_intra,num_site, sites) 

    L = Int(log2(num_site))
    break_mpo = break_chain(L_chain, num_site, sites)
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
    k_cen = apply(hop_intra,k_mpo_1)
    true_hop_1 = apply(break_mpo, k_cen)

    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
    k_cen = apply( k_mpo_2,hop_intra)
    true_hop_2 = apply(k_cen, break_mpo)
    
    k_mpo =  +(true_hop_1, true_hop_2;  cutoff = 1e-8)  

    return k_mpo 
end

# shift the off-diag line

function arbitarty_offline(k_mpo,demand_order)
    k_mpo_o1 = k_mpo
    k_mpo_o2 = apply(k_mpo,k_mpo)
    target_mpo = k_mpo
    for iter_num in 1:demand_order
        if iter_num  == 1
            target_mpo = k_mpo_o1
        elseif iter_num  == 2
            target_mpo = k_mpo_o2
        else
            target_mpo = apply(k_mpo, k_mpo_o2)
            k_mpo_o2 = target_mpo
        end
    end
    return target_mpo
end

function exponential_shift(k_mpo,demand_order)
    #2^L shifting creation
    target_mpo_ini = apply(k_mpo,k_mpo)
    target_mpo = target_mpo_ini
    for iter_num in 1:demand_order+1
        if iter_num  == 1
            target_mpo = k_mpo 
        elseif iter_num  == 2
            target_mpo = target_mpo_ini
        else
            target_mpo = apply(target_mpo,target_mpo)
        end
    end
    return target_mpo
end

function to_binary_vector(n, size)
    # Convert to binary string
    binary_str = string(n, base=2)
    
    # Pad the binary string with leading zeros to match the desired size
    padded_binary_str = lpad(binary_str, size, '0')
    
    # Convert the padded string into a vector of strings (each character is a string)
    return collect(padded_binary_str) |> x -> map(s -> string(s), x)
end

#for validating of small system Hamiltonian
function get_matrix(mpo,size )
    mat = zeros(size, size) 
    for i in 0:size-1
        for j in 0:size-1
            element = inner(randomMPS(sites,to_binary_vector(Int(i),Int(log2(size))))',mpo,randomMPS(sites,to_binary_vector(Int(j),Int(log2(size)))))
            mat[i+1,j+1] = element
        end
    end
    return mat
end
#######################################

#2. For square lattice
################################
function interchain_hopping_square(L_chain,hop_inter, num_site, sites) 
    L = Int(log2(num_site))
     
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
    #shift it
    shifting_order = Int(log2(L_chain))

    K_mpo_1_true = exponential_shift(k_mpo_1, shifting_order) 
    k_mpo_1_t = apply(hop_inter,K_mpo_1_true)
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
    #shift it
    K_mpo_2_ture =  exponential_shift(k_mpo_2, shifting_order) 
    k_mpo_2_t = apply(K_mpo_2_ture,hop_inter)
    
    
    k_mpo = k_mpo_1_t + k_mpo_2_t 

    return k_mpo 
end
##############################




#3. honeycomb
function interchain_hopping_honeycomb(L_chain, num_site,inter_hop1,inter_hop2, sites) 
    L = Int(log2(num_site))
   
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
    
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
   
    shifting_order = Int(log2(L_chain))

    shift_mpo_up = exponential_shift(k_mpo_1, shifting_order) 
    shift_mpo_dn = exponential_shift(k_mpo_2, shifting_order) 

    hop_up_1 = apply(k_mpo_1, shift_mpo_up)
    hop_up_2 = apply(shift_mpo_up,k_mpo_2)
    hop_dn_1 = apply(shift_mpo_dn,k_mpo_2)
    hop_dn_2 = apply(k_mpo_1,shift_mpo_dn)
    
    real_hop_up_1 = apply(inter_hop1,  hop_up_1)
    real_hop_up_2 = apply(inter_hop2, hop_up_2)
    real_hop_dn_1 = apply( hop_dn_1,inter_hop1)
    real_hop_dn_2 = apply( hop_dn_2,inter_hop2)
    
    k_mpo =  real_hop_up_1 + real_hop_dn_1 + real_hop_dn_2 + real_hop_up_2 

    return k_mpo 
end
