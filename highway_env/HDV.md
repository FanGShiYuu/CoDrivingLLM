# Background Vehicle Model: IDM and MOBIL Models

## Intelligent Driver Model (IDM)

The acceleration of the HDV is given by the IDM as shown below:

$$
a_f(s,v,\Delta v) = a_{\max}\left[1-\left(\frac{v}{v_{\mathrm{d}}}\right)^\delta-\left(\frac{s^*(v,\Delta v)}{s}\right)^2\right]
$$

$$
s^*(v,\Delta v) = s_0 + v T_g + \frac{v \Delta v}{2\sqrt{a_{\max}a_{dd}}}
$$

Where:  
- $a_f(s,v,\Delta v)$: acceleration derived from IDM  
- $v_{\mathrm{d}}$: desired velocity  
- $\delta$: acceleration exponent  
- $\Delta v$: velocity difference between the front vehicle and the subject vehicle  
- $s$: distance between the front vehicle and the subject vehicle  
- $s^*(v,\Delta v)$: expected distance  
- $s_0$: minimum stopping distance  
- $T_g$: desired time gap  
- $a_{\max}$: maximum acceleration  
- $a_{dd}$: desired deceleration  

## Minimizing Overall Braking Induced by Lane changes (MOBIL Model)

Furthermore, MOBIL achieves safe and efficient traffic flow by minimizing the overall braking caused by lane changes, which mainly includes two parts: **lane change incentive** and **safety inspection**.  

### Lane Change Incentive

The lane change incentive evaluates the change in the acceleration of the ego-vehicle and surrounding vehicles to determine whether a lane change is warranted:

$$
a_{c,old}-a_{c,new} + p(a_{n,old}-a_{n,new} + a_{o,old}-a_{o,new}) \geq \Delta a_{\text{th}}
$$

Where:  
- $a_{c,old}$ and $a_{c,new}$: acceleration of vehicles before and after the lane change
- c, n, and o: ego vehicle, new follower, and old follower
- $p$: politeness coefficient, indicating the attention given to surrounding vehicles  
- $\Delta a_{\text{th}}$: acceleration gain required to trigger a lane change  

### Safety Inspection

In order to ensure the safety of a lane change, the MOBIL model carries out a safety inspection to ensure that the lane change will not cause a sudden brake on the rear vehicle of the target lane:

$$
a_n \geq -b_{\text{safe}}
$$

Where $b_{\text{safe}}$ is the maximum braking imposed on a vehicle during a cut-in.

