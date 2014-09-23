set terminal pngcairo size 1200,800 

set title 'ololo' 
set output 'result.png'
plot 'result_00' using 1:2 lw 2 with points, 'result_01' using 1:2 lw 2 with points
