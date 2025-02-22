// Copyright © 2023–2025 Dario Izzo (dario.izzo@gmail.com), 
// Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the kep3 library.
//
// Licensed under the Mozilla Public License, version 2.0.
// You may obtain a copy of the MPL at https://www.mozilla.org/MPL/2.0/.

#ifndef kep3_IC2PAR2IC_H
#define kep3_IC2PAR2IC_H

#include <array>

#include <kep3/detail/visibility.hpp>

namespace kep3
{
kep3_DLL_PUBLIC std::array<double, 6> ic2par(const std::array<std::array<double, 3>, 2> &pos_vel, double mu);

kep3_DLL_PUBLIC std::array<std::array<double, 3>, 2> par2ic(const std::array<double, 6> &par, double mu);
} // namespace kep3
#endif // kep3_IC2PAR2IC_H