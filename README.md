# Introduction
The skip tracker is a simple code to track number of skips. You are provided three set of windows which shows realtime image data and how its being tracked. Certain assumptions are made such as legs are the only skin tone object and the backgroud is of different color.
For detecting skips in real time, this uses a buffer of 20 previous data and a lower peak is also calcualted to seperate actuall skips from noise  

# Why 
Well its hard to keep track of your skips and mentaly once you reach a number you thought was not possible, you tend to feel tired. So not knowing the number lets you push you to your limit. Just give it all you got and see what you achived


# Learnings
It was a nice weekend project, Learnings were
>- Python open cv. Going through other similar problems such as hand detection and well fun of reading documentation  
>- Read multiple papers on real time peek detection. 

# Refrence
[![Really good explanation on peak detection ]()](
https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data ) 