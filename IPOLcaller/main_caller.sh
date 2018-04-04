#!/bin/sh

#main_caller.sh input_0.png input_1.png 7 $orsa $desc_type $optimal_affine_simu $default_tilt_tol $radius $acontrario $default_matching_ratio $match_ratio $eigen_thresh

#${demoextras}/main_caller.sh input_0.png input_1.png $apply_orsa $orsa $desc_type $optimal_affine_simu $default_tilt_tol $radius $acontrario $default_matching_ratio $match_ratio $default_orsa_precision $orsa_precision $apply_mult2one $eigen $eigen_thresh $filter_radius

#filepath=$(pwd)
#parentname="$(basename "$(dirname "$filepath")")"
#path="../../../../ipol_demo/modules/demorunner/binaries/$parentname/bin"
toexec="${bin}main -im1 $1 -im2 $2 -desc $5 -applyfilter $4"

#$optimal_affine_simu
if [ "$6" = "True" ] 2>/dev/null; then
	#$default_tilt_tol
	if [ "$7" = "False" ] 2>/dev/null; then
		toexec="$toexec -covering $8"
	fi
else
	toexec="$toexec -covering 1"
fi

#$acontrario
if [ "$9" = "True" ] 2>/dev/null; then
	if [ ! -f input_2.png ]; then
		cp "${demoextras}/im3_sub.png" "input_2.png"
	fi
	toexec="$toexec -im3 input_2.png"
fi


#$default_matching_ratio
if [ "${10}" = "False" ] 2>/dev/null; then
	toexec="$toexec -match_ratio ${11}"
fi

#$default_orsa_precision $orsa_precision
if [ "${12}" = "False" ] 2>/dev/null; then
	toexec="$toexec -filter_precision ${13}"
fi

#eigen $eigen_thresh
if [ "${15}" = "False" ] 2>/dev/null; then
	toexec="$toexec -eigen_threshold ${16}"
fi

#tensoreigen $tensor_eigen_thresh
if [ "${17}" = "False" ] 2>/dev/null; then
	toexec="$toexec -tensor_eigen_threshold ${18}"
fi


toexec="$toexec -filter_radius ${19}"

if [ "${20}" = "True" ] 2>/dev/null; then
	toexec="$toexec -fixed_area"
fi

#echo "main_caller.sh has been called with :"
#echo "$@"
#echo " ${11}"
#echo "$toexec"
#echo " "
#OMP_NUM_THREADS=5 $toexec
$toexec

#echo "var=1234" > algo_info.txt
