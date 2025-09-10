// Copyright © 2023–2025 Dario Izzo (dario.izzo@gmail.com),
// Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the kep3 library.
//
// Licensed under the Mozilla Public License, version 2.0.
// You may obtain a copy of the MPL at https://www.mozilla.org/MPL/2.0/.

#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/math/select.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/taylor.hpp>

#include <kep3/core_astro/constants.hpp>
#include <kep3/ta/stark_cr3bp.hpp>

using heyoka::eq;
using heyoka::expression;
using heyoka::make_vars;
using heyoka::par;
using heyoka::pow;
using heyoka::prime;
using heyoka::sqrt;
using heyoka::sum;
using heyoka::taylor_adaptive;
using heyoka::var_ode_sys;

namespace kep3::ta
{
std::tuple<std::vector<std::pair<expression, expression>>, expression, expression> controlled_expression_factory()
{
    // The symbolic variables.
    auto [x, y, z, vx, vy, vz, m] = make_vars("x", "y", "z", "vx", "vy", "vz", "m");

    // Renaming parameters.
    const auto &mu = par[0];
    const auto &veff = par[1];
    const auto &[ux, uy, uz] = std::array{par[2], par[3], par[4]};

    // Distances to the bodies.
    auto r_1 = sqrt(sum({pow(x + mu, 2.), pow(y, 2.), pow(z, 2.)}));
    auto r_2 = sqrt(sum({pow(x - (1. - mu), 2.), pow(y, 2.), pow(z, 2.)}));

    // The thrust magnitude
    const auto u_norm = sqrt(sum({pow(ux, 2.), pow(uy, 2.), pow(uz, 2.)}));

    // The Equations of Motion.
    const auto xdot = vx;
    const auto ydot = vy;
    const auto zdot = vz;
    const auto vxdot = 2. * vy + x - (1. - mu) * (x + mu) / (pow(r_1, 3.)) - mu * (x + mu - 1.) / pow(r_2, 3.) + ux / m;
    const auto vydot = -2. * vx + y - (1. - mu) * y / pow(r_1, 3.) - mu * y / pow(r_2, 3.) + uy / m;
    const auto vzdot = -(1. - mu) * z / pow(r_1, 3.) - mu * z / pow(r_2, 3.) + uz / m;
    
    // To avoid singularities in the corner case u_norm=0. we use a select here. Implications on performances should be
    // studied.
    const auto mdot = select(eq(u_norm, 0.), 0., -u_norm / veff);
    
    // The effective potential. (note the sign convention here)
    const auto U = 1. / 2. * (pow(x, 2.) + pow(y, 2.)) + (1. - mu) / r_1 + mu / r_2;
    // The velocity squared (in rotating).
    const auto v2 = (pow(vx, 2.) + pow(vy, 2.) + pow(vz, 2.));
    // The Jacobi constant.
    const auto C = 2. * U - v2;

    return {
        {prime(x) = xdot, prime(y) = ydot, prime(z) = zdot, prime(vx) = vxdot, prime(vy) = vydot, prime(vz) = vzdot, prime(m) = mdot},
        U,
        C};
};

std::vector<std::pair<expression, expression>> stark_cr3bp_dyn()
{
    return std::get<0>(controlled_expression_factory());
}

heyoka::expression stark_cr3bp_effective_potential_U()
{
    return std::get<1>(controlled_expression_factory());
}

heyoka::expression stark_cr3bp_jacobi_C()
{
    return std::get<2>(controlled_expression_factory());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex ta_stark_cr3bp_mutex;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<double, taylor_adaptive<double>> ta_stark_cr3bp_cache;

const heyoka::taylor_adaptive<double> &get_ta_stark_cr3bp(double tol)
{
    // Lock down for access to cache.
    std::lock_guard const lock(ta_stark_cr3bp_mutex);

    // Lookup.
    if (auto it = ta_stark_cr3bp_cache.find(tol); it == ta_stark_cr3bp_cache.end()) {
        // Cache miss, create new one.
        const std::vector init_state = {1., 1., 1., 1., 1., 1., 1.};
        auto new_ta = taylor_adaptive<double>{stark_cr3bp_dyn(), init_state, heyoka::kw::tol = tol};
        return ta_stark_cr3bp_cache.insert(std::make_pair(tol, std::move(new_ta))).first->second;
    } else {
        // Cache hit, return existing.
        return it->second;
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex ta_stark_cr3bp_var_mutex;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<double, taylor_adaptive<double>> ta_stark_cr3bp_var_cache;

const heyoka::taylor_adaptive<double> &get_ta_stark_cr3bp_var(double tol)
{
    // Lock down for access to cache.
    std::lock_guard const lock(ta_stark_cr3bp_var_mutex);

    // Lookup.
    if (auto it = ta_stark_cr3bp_var_cache.find(tol); it == ta_stark_cr3bp_var_cache.end()) {
        auto [x, y, z, vx, vy, vz, m] = make_vars("x", "y", "z", "vx", "vy", "vz", "m");
        auto vsys = var_ode_sys(stark_cr3bp_dyn(), {x, y, z, vx, vy, vz, m}, 1);
        // Cache miss, create new one.
        const std::vector init_state = {1., 1., 1., 1., 1., 1., 1.};
        auto new_ta = taylor_adaptive<double>{vsys, init_state, heyoka::kw::tol = tol, heyoka::kw::compact_mode = true};
        return ta_stark_cr3bp_var_cache.insert(std::make_pair(tol, std::move(new_ta))).first->second;
    } else {
        // Cache hit, return existing.
        return it->second;
    }
}

size_t get_ta_stark_cr3bp_cache_dim()
{
    // Lock down for access to cache.
    std::lock_guard const lock(ta_stark_cr3bp_mutex);
    return ta_stark_cr3bp_cache.size();
}

size_t get_ta_stark_cr3bp_var_cache_dim()
{
    // Lock down for access to cache.
    std::lock_guard const lock(ta_stark_cr3bp_mutex);
    return ta_stark_cr3bp_var_cache.size();
}

} // namespace kep3::ta