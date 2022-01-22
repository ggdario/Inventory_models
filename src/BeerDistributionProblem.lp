\* EOQ_No_Setup *\
Minimize
Sum_of_Transporting_Costs: 9 Route_O1_1 + 9.1 Route_O1_2 + 9.2 Route_O1_3
 + 9.3 Route_O1_4 + 100000 Route_O2_1 + 9 Route_O2_2 + 9.1 Route_O2_3
 + 9.2 Route_O2_4 + 100000 Route_O3_1 + 100000 Route_O3_2 + 9 Route_O3_3
 + 9.1 Route_O3_4 + 100000 Route_O4_1 + 100000 Route_O4_2 + 100000 Route_O4_3
 + 9 Route_O4_4 + 6 Route_R1_1 + 6.1 Route_R1_2 + 6.2 Route_R1_3
 + 6.3 Route_R1_4 + 100000 Route_R2_1 + 6 Route_R2_2 + 6.1 Route_R2_3
 + 6.2 Route_R2_4 + 100000 Route_R3_1 + 100000 Route_R3_2 + 6 Route_R3_3
 + 6.1 Route_R3_4 + 100000 Route_R4_1 + 100000 Route_R4_2 + 100000 Route_R4_3
 + 6 Route_R4_4
Subject To
Sum_of_Products_into_Bar1: Route_O1_1 + Route_O2_1 + Route_O3_1 + Route_O4_1
 + Route_R1_1 + Route_R2_1 + Route_R3_1 + Route_R4_1 >= 100
Sum_of_Products_into_Bar2: Route_O1_2 + Route_O2_2 + Route_O3_2 + Route_O4_2
 + Route_R1_2 + Route_R2_2 + Route_R3_2 + Route_R4_2 >= 190
Sum_of_Products_into_Bar3: Route_O1_3 + Route_O2_3 + Route_O3_3 + Route_O4_3
 + Route_R1_3 + Route_R2_3 + Route_R3_3 + Route_R4_3 >= 210
Sum_of_Products_into_Bar4: Route_O1_4 + Route_O2_4 + Route_O3_4 + Route_O4_4
 + Route_R1_4 + Route_R2_4 + Route_R3_4 + Route_R4_4 >= 160
Sum_of_Products_into_BarE: Route_O1_E + Route_O2_E + Route_O3_E + Route_O4_E
 + Route_R1_E + Route_R2_E + Route_R3_E + Route_R4_E >= 20
Sum_of_Products_out_of_Warehouse_O1: Route_O1_1 + Route_O1_2 + Route_O1_3
 + Route_O1_4 + Route_O1_E <= 50
Sum_of_Products_out_of_Warehouse_O2: Route_O2_1 + Route_O2_2 + Route_O2_3
 + Route_O2_4 + Route_O2_E <= 60
Sum_of_Products_out_of_Warehouse_O3: Route_O3_1 + Route_O3_2 + Route_O3_3
 + Route_O3_4 + Route_O3_E <= 80
Sum_of_Products_out_of_Warehouse_O4: Route_O4_1 + Route_O4_2 + Route_O4_3
 + Route_O4_4 + Route_O4_E <= 70
Sum_of_Products_out_of_Warehouse_R1: Route_R1_1 + Route_R1_2 + Route_R1_3
 + Route_R1_4 + Route_R1_E <= 90
Sum_of_Products_out_of_Warehouse_R2: Route_R2_1 + Route_R2_2 + Route_R2_3
 + Route_R2_4 + Route_R2_E <= 100
Sum_of_Products_out_of_Warehouse_R3: Route_R3_1 + Route_R3_2 + Route_R3_3
 + Route_R3_4 + Route_R3_E <= 120
Sum_of_Products_out_of_Warehouse_R4: Route_R4_1 + Route_R4_2 + Route_R4_3
 + Route_R4_4 + Route_R4_E <= 110
Bounds
 0 <= Route_O1_1
 0 <= Route_O1_2
 0 <= Route_O1_3
 0 <= Route_O1_4
 0 <= Route_O1_E
 0 <= Route_O2_1
 0 <= Route_O2_2
 0 <= Route_O2_3
 0 <= Route_O2_4
 0 <= Route_O2_E
 0 <= Route_O3_1
 0 <= Route_O3_2
 0 <= Route_O3_3
 0 <= Route_O3_4
 0 <= Route_O3_E
 0 <= Route_O4_1
 0 <= Route_O4_2
 0 <= Route_O4_3
 0 <= Route_O4_4
 0 <= Route_O4_E
 0 <= Route_R1_1
 0 <= Route_R1_2
 0 <= Route_R1_3
 0 <= Route_R1_4
 0 <= Route_R1_E
 0 <= Route_R2_1
 0 <= Route_R2_2
 0 <= Route_R2_3
 0 <= Route_R2_4
 0 <= Route_R2_E
 0 <= Route_R3_1
 0 <= Route_R3_2
 0 <= Route_R3_3
 0 <= Route_R3_4
 0 <= Route_R3_E
 0 <= Route_R4_1
 0 <= Route_R4_2
 0 <= Route_R4_3
 0 <= Route_R4_4
 0 <= Route_R4_E
Generals
Route_O1_1
Route_O1_2
Route_O1_3
Route_O1_4
Route_O1_E
Route_O2_1
Route_O2_2
Route_O2_3
Route_O2_4
Route_O2_E
Route_O3_1
Route_O3_2
Route_O3_3
Route_O3_4
Route_O3_E
Route_O4_1
Route_O4_2
Route_O4_3
Route_O4_4
Route_O4_E
Route_R1_1
Route_R1_2
Route_R1_3
Route_R1_4
Route_R1_E
Route_R2_1
Route_R2_2
Route_R2_3
Route_R2_4
Route_R2_E
Route_R3_1
Route_R3_2
Route_R3_3
Route_R3_4
Route_R3_E
Route_R4_1
Route_R4_2
Route_R4_3
Route_R4_4
Route_R4_E
End
