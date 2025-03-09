// Copyright © 2023–2025 Dario Izzo (dario.izzo@gmail.com),
// Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the kep3 library.
//
// Licensed under the Mozilla Public License, version 2.0.
// You may obtain a copy of the MPL at https://www.mozilla.org/MPL/2.0/.

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <kep3/core_astro/constants.hpp>
#include <kep3/core_astro/mima.hpp>
#include <kep3/lambert_problem.hpp>

#include "catch.hpp"

TEST_CASE("mima")
{
    // We take the first item from the zeonodo database https://zenodo.org/records/11502524 containing
    // independent cases reporting the mima values.
    {
        std::array<double, 3> rs = {3.574644002632926178e+10, -5.688222150272903442e+10, -1.304897435568400574e+10};
        std::array<double, 3> vs = {4.666425901145393436e+04, 2.375697019573154466e+04, 1.165422004315219965e+04};
        std::array<double, 3> rt = {7.672399994418635559e+10, -1.093562401274179382e+11, 4.796635567684053421e+09};
        std::array<double, 3> vt = {2.725105661271001009e+04, 1.599599495457483499e+04, 6.818440757625087826e+03};
        double tof = 3.311380772794449854e+02 * kep3::DAY2SEC;
        double Tmax = 0.6;
        double veff = kep3::G0 * 4000;
        auto lp = kep3::lambert_problem(rs, rt, tof, kep3::MU_SUN);
        std::array<double, 3> dv1 = {lp.get_v0()[0][0] - vs[0], lp.get_v0()[0][1] - vs[1], lp.get_v0()[0][2] - vs[2]};
        std::array<double, 3> dv2 = {vt[0] - lp.get_v1()[0][0], vt[1] - lp.get_v1()[0][1], vt[2] - lp.get_v1()[0][2]};
        auto mima_res = kep3::mima(dv1, dv2, tof, Tmax, veff);
        double mima_from_zenodo_db = 1.711341975126993020e+02;
        REQUIRE(mima_res.first == Approx(mima_from_zenodo_db).epsilon(1e-8));
    }
    // We take a second item from the zeonodo database https://zenodo.org/records/11502524
    {
        std::array<double, 3> rs = {-4.054163103119991455e+11, -3.112051036509102173e+11, 8.852823556219964600e+10};
        std::array<double, 3> vs = {1.053626627938712227e+04, -9.040187399709659076e+03, 6.140398196916326924e+03};
        std::array<double, 3> rt = {-4.377345423917691040e+10, -4.913367837977642822e+11, 3.030052465928871918e+10};
        std::array<double, 3> vt = {1.549115173657366176e+04, -3.341220883615214916e+02, -3.245198147308494299e+03};
        double tof = 3.219932820383384069e+02 * kep3::DAY2SEC;
        double Tmax = 0.6;
        double veff = kep3::G0 * 4000;
        auto lp = kep3::lambert_problem(rs, rt, tof, kep3::MU_SUN);
        std::array<double, 3> dv1 = {lp.get_v0()[0][0] - vs[0], lp.get_v0()[0][1] - vs[1], lp.get_v0()[0][2] - vs[2]};
        std::array<double, 3> dv2 = {vt[0] - lp.get_v1()[0][0], vt[1] - lp.get_v1()[0][1], vt[2] - lp.get_v1()[0][2]};
        auto mima_res = kep3::mima(dv1, dv2, tof, Tmax, veff);
        double mima_from_zenodo_db = 1.101217159182178875e+03;
        REQUIRE(mima_res.first == Approx(mima_from_zenodo_db).epsilon(1e-8));
    }
}
