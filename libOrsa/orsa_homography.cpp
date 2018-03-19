/**
 * @file orsa_homography.cpp
 * @brief Homographic image registration
 * @author Pascal Monasse, Pierre Moulon
 * 
 * Copyright (c) 2011-2012 Pascal Monasse, Pierre Moulon
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdlib>
#include <ctime>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "match.hpp"


#include "libOrsa/homography_model.hpp"

#include "../libNumerics/numerics.h"
#include "../libSimuTilts/library.h"


namespace orsa {
/// Number of random samples in ORSA

bool TransformH(const libNumerics::matrix<double> &H, double &x, double &y)
{
  libNumerics::vector<double> X(3);
  X(0)=x; X(1)=y; X(2)=1.0;
  X = H*X;
  bool positive = (X(2)*H(2,2)>0);
  X /= X(2);
  x = X(0); y = X(1);
  return positive;
}


/// Display average/max error of inliers of homography H.
static void display_stats(const std::vector<Match>& vec_matchings,
                          const std::vector<int>& vec_inliers,
                          libNumerics::matrix<double>& H,bool verb) {
  std::vector<int>::const_iterator it=vec_inliers.begin();
  double l2=0, linf=0;
  for(; it!=vec_inliers.end(); ++it) {
    const Match& m=vec_matchings[*it];
    double x1=m.x1, y1=m.y1;
    TransformH(H, x1, y1);
    double e = (m.x2-x1)*(m.x2-x1) + (m.y2-y1)*(m.y2-y1);
    l2 += e;
    if(linf < e)
      linf = e;
  }
  if (verb)
      std::cout << "Average/max error: "
                << sqrt(l2/vec_inliers.size()) << "/"
                << sqrt(linf) <<std::endl;
}

/// ORSA homography estimation
bool ORSA_homography(const std::vector<Match>& vec_matchings, int w1,int h1, int w2,int h2,
          double precision, int nbIter, libNumerics::matrix<double>& H, std::vector<int>& vec_inliers,double& nfa, bool verb)
{
  const int n = static_cast<int>( vec_matchings.size() );
  if(n < 5)
  {
      if (verb)
          std::cerr << "Error: ORSA Homography needs 5 matches or more to proceed" <<std::endl;
      return false;
  }
  libNumerics::matrix<double> xA(2,n), xB(2,n);

  for (int i=0; i < n; ++i)
  {
    xA(0,i) = vec_matchings[i].x1;
    xA(1,i) = vec_matchings[i].y1;
    xB(0,i) = vec_matchings[i].x2;
    xB(1,i) = vec_matchings[i].y2;
  }

  orsa::HomographyModel model(xA, w1, h1, xB, w2, h2, true);
  //model.setConvergenceCheck(true);
  nfa = model.orsa(vec_inliers, nbIter, &precision, &H, verb);
  if(nfa>0.0)
    return false;
  std::cout << "Before refinement: ";
  display_stats(vec_matchings, vec_inliers, H, verb);
  if( model.ComputeModel(vec_inliers,&H) ) // Re-estimate with all inliers
  {
    if (verb)
        std::cout << "After  refinement: ";
    display_stats(vec_matchings, vec_inliers, H, verb);
  } else
  {
    if (verb)
        std::cerr << "Warning: error in refinement, result is suspect" <<std::endl;
  }
  return true;
}

} //namespace orsa
