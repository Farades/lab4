set terminal pngcairo size 1200,800

set output 'result_forest_H.png'
plot 'result_forest' using 1:2 lw 2 with points

set output 'result_forest_V.png'
plot 'result_forest' using 3:4 lw 2 with points

set output 'result_road_H.png'
plot 'result_road' using 1:2 lw 2 with points

set output 'result_road_V.png'
plot 'result_road' using 3:4 lw 2 with points
