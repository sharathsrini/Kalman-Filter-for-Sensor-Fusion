
## Kalman Filter for Sensor Fusion

## Idea Of ​​The Kalman Filter In A Single-Dimension

Kalman filters are discrete systems that allows us to define a dependent variable by an independent variable, where by we will solve for the independent variable so that when we are given measurements (the dependent variable),we can infer an estimate     of the independent variable assuming that noise exists from our input measurement and noise also exists in how we’ve modeled the world with our math equations because of inevitably unaccounted for factors in the non-sterile world.Input variables become more valuable when modeled as a system of equations,ora  matrix, in order to make it possible to determine the relationships between those values. Every variables in every dimension will contain noise, and therefore the introduction of related inputs will allow weighted averaging to take place based on the predicted differential at the next step, the noise unaccounted for in the system,and the noise introduced by the sensor inputs.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sb
from scipy import stats
import time
```


```python
%matplotlib inline
fw = 10 # figure width
```

##  *Despite noisy measurement of individual sensors, We can calculate an optimal estimate of all conditions*.

### https://in.udacity.com/course/artificial-intelligence-for-robotics--cs373

#### Plot the Distributions in this range:


```python
x = np.linspace(-100,100,1000)
```


```python
mean0 = 0.0   # e.g. meters or miles
var0  = 20.0
```


```python
plt.figure(figsize=(fw,5))
plt.plot(x,mlab.normpdf(x, mean0, var0), label='Normal Distribution')
plt.ylim(0, 0.1);
plt.legend(loc='best');
plt.xlabel('Position');
```


![png](output_9_0.png)


## Now we have something, which estimates the moved distance

#### The Mean is  meters, calculated from velocity*dt or step counter or wheel encoder ...

#### VarMove is the Estimated or determined with static measurements


```python
meanMove = 25.0
varMove  = 10.0 
```


```python
plt.figure(figsize=(fw,5))
plt.plot(x,mlab.normpdf(x, meanMove, varMove), label='Normal Distribution')
plt.ylim(0, 0.1);
plt.legend(loc='best');
plt.xlabel('Distance moved');
```


![png](output_13_0.png)


Both Distributions have to be merged together
$\mu_\text{new}=\mu_\text{0}+\mu_\text{move}$ is the new mean and $\sigma^2_\text{new}=\sigma^2_\text{0}+\sigma^2_\text{move}$ is the new variance.




```python
def predict(var, mean, varMove, meanMove):
    new_var = var + varMove
    new_mean= mean+ meanMove
    return new_var, new_mean
```


```python
new_var, new_mean = predict(var0, mean0, varMove, meanMove)
```


```python
plt.figure(figsize=(fw,5))
plt.plot(x,mlab.normpdf(x, mean0, var0), label='Beginning Normal Distribution')
plt.plot(x,mlab.normpdf(x, meanMove, varMove), label='Movement Normal Distribution')
plt.plot(x,mlab.normpdf(x, new_mean, new_var), label='Resulting Normal Distribution')
plt.ylim(0, 0.1);
plt.legend(loc='best');
plt.title('Normal Distributions of 1st Kalman Filter Prediction Step');
plt.savefig('Kalman-Filter-1D-Step.png', dpi=150)
```


![png](output_17_0.png)


### What you see: The resulting distribution is flat > uncertain.

The more often you run the predict step, the flatter the distribution get

First Sensor Measurement (Position) is coming in...
#### Sensor Defaults for Position Measurements
(Estimated or determined with static measurements)


```python
meanSensor = 25.0
varSensor  = 12.0
```


```python
plt.figure(figsize=(fw,5))
plt.plot(x,mlab.normpdf(x, meanSensor, varSensor))
plt.ylim(0, 0.1);
```


![png](output_20_0.png)


Now both Distributions have to be merged together
$\sigma^2_\text{new}=\cfrac{1}{\cfrac{1}{\sigma^2_\text{old}}+\cfrac{1}{\sigma^2_\text{Sensor}}}$ is the new variance and the new mean value is $\mu_\text{new}=\cfrac{\sigma^2_\text{Sensor} \cdot \mu_\text{old} + \sigma^2_\text{old} \cdot \mu_\text{Sensor}}{\sigma^2_\text{old}+\sigma^2_\text{Sensor}}$


```python
def correct(var, mean, varSensor, meanSensor):
    new_mean=(varSensor*mean + var*meanSensor) / (var+varSensor)
    new_var = 1/(1/var +1/varSensor)
    return new_var, new_mean
```


```python
var, mean = correct(new_var, new_mean, varSensor, meanSensor)
```


```python
plt.figure(figsize=(fw,5))
plt.plot(x,mlab.normpdf(x, new_mean, new_var), label='Beginning (after Predict)')
plt.plot(x,mlab.normpdf(x, meanSensor, varSensor), label='Position Sensor Normal Distribution')
plt.plot(x,mlab.normpdf(x, mean, var), label='New Position Normal Distribution')
plt.ylim(0, 0.1);
plt.legend(loc='best');
plt.title('Normal Distributions of 1st Kalman Filter Update Step');
```


![png](output_24_0.png)


###### This is called the Measurement or Correction step! The Filter get's more serious about the actual state.

#### Let's put everything together: The 1D Kalman Filter
"Kalman-Filter: Predicting the Future since 1960"

Let's say, we have some measurements for position and for distance traveled. Both have to be fused with the 1D-Kalman Filter.


```python
positions = (10, 20, 30, 40, 50)+np.random.randn(5)
distances = (10, 10, 10, 10, 10)+np.random.randn(5)
```


```python
positions
```




    array([10.90899944, 19.59541839, 30.51879108, 40.06235696, 50.1309251 ])




```python
distances
```




    array([ 9.74429558,  9.78301188, 11.77885122, 10.40314194, 10.92776063])




```python
for m in range(len(positions)):
    
    # Predict
    var, mean = predict(var, mean, varMove, distances[m])
    #print('mean: %.2f\tvar:%.2f' % (mean, var))
    plt.plot(x,mlab.normpdf(x, mean, var), label='%i. step (Prediction)' % (m+1))
    
    # Correct
    var, mean = correct(var, mean, varSensor, positions[m])
    print('After correction:  mean= %.2f\tvar= %.2f' % (mean, var))
    plt.plot(x,mlab.normpdf(x, mean, var), label='%i. step (Correction)' % (m+1))
    
plt.ylim(0, 0.1);
plt.xlim(-20, 120)
plt.legend();
```

    After correction:  mean= 20.26	var= 7.29
    After correction:  mean= 23.88	var= 7.08
    After correction:  mean= 32.64	var= 7.05
    After correction:  mean= 41.29	var= 7.04
    After correction:  mean= 50.99	var= 7.04
    


![png](output_30_1.png)



The sensors are represented as normal distributions with their parameters ($\mu$ and $\sigma^2$) and are calculated together with addition or convolution. The prediction decreases the certainty about the state, the correction increases the certainty.

Prediction: Certainty $\downarrow$
Correction: Certainty $\uparrow$

## Kalman Filter - Multi-Dimensional Measurement

#### Multidimensional Kalman filter


Let's assume we drive our car  into a tunnel. The GPS signal is gone. Nevertheless, we might want to get notified that should  exit in the tunnel.The procedure is using the example of a vehicle with navigation device, which enters a tunnel. The last known position is before losing the GPS signal. Afterwards (with permanently installed Navis) only the speed information of the vehicle (wheel speeds & yaw rate) is available as normal distributed noisy measured variable. From this a velocity in x and y can be calculated. 

#### *How would we know now where we are right now? 

##### *It merges the vehicle sensors and calculates the position as well as possible.*



Now lets think, when we were at the tunnel entrance last and drive at 50km / h, then the car can indeed calculated exactly where (x = position) you are 1 minute (t = time) later


So far the perfect world. But the calculation takes over a microcontroller and this relies on sensors. Both the sensors have random errors, the transmission path has interference, and the resolution of CAN bus or analog-to-digital converters can cause many inaccuracies in the simple statement "speed". For example, a speed signal looks like this:

Speed-time course of a measurement
Speed-time course of a measurement

On average, the measured speed is already correct, but there is some "noise". If one calculates a histogram of the determined speeds, one sees that the determined values ​​are approximately subject to a normal distribution.

Histogram of measured velocity with normal distribution
Histogram of measured velocity with normal distribution

So there is one, and really only one, maximum value (unimodal) and a spread (variance). If this is the case, you can do the calculation very well with a trick nevertheless.




### State Vector
Constant Velocity Model for Ego Motion

$$x_k= \left[ \matrix{ x \\ y \\ \dot x \\ \dot y} \right] = \matrix{ \text{Position X} \\ \text{Position Y} \\ \text{Velocity in X} \\ \text{Velocity in Y}}$$



```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
print(x, x.shape)
plt.scatter(float(x[0]),float(x[1]), s=100)
plt.title('Initial Location')
```

    [[0.]
     [0.]
     [0.]
     [0.]] (4, 1)
    




    <matplotlib.text.Text at 0x1c573664f60>




![png](output_35_2.png)



```python
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
print(P, P.shape)

```

    [[1000.    0.    0.    0.]
     [   0. 1000.    0.    0.]
     [   0.    0. 1000.    0.]
     [   0.    0.    0. 1000.]] (4, 4)
    

## Using Matplotlib to understand the P matrix More:

The value of the vectors on the x,y and $\dot x$, $\dot y$ is very high. Thus the intial position and velocity is very uncertain

While plotting the matrix, make sure we label:
1. Setting the locations of the yticks
2. Setting the locations and labels of the yticks
3. Setting the locations of the yticks
4. Setting the locations and labels of the yticks


## Covariance matrix $P$

An uncertainty must be given for the initial state  x0 . In the 1D case, the σ0 , now a matrix, defines an initial uncertainty for all states.

This matrix is ​​most likely to be changed during the filter passes. It is changed in both the Predict and Correct steps. If one is quite sure about the states at the beginning, one can use low values ​​here, if one does not know exactly how the values ​​of the state vector are, the covariance matrix should be Pinitialized with very large values ​​(1 million or so) to allow the filter to converge relatively quickly (find the right values ​​based on the measurements).





```python
fig = plt.figure(figsize=(6, 6))
im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Initial Covariance Matrix $P$')
ylocs, ylabels = plt.yticks()
plt.yticks(np.arange(7))
# 
plt.yticks(np.arange(6),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)
xlocs, xlabels = plt.xticks()
# 
plt.xticks(np.arange(7))

plt.xticks(np.arange(6),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

plt.xlim([-0.5,3.5])
plt.ylim([3.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax);

```


![png](output_38_0.png)


#### Dynamics matrix A
The core of the filter, however, is the following definition, which we would set up yourself only with great understanding of the physical context. This is not easy for many real problems. For our simple example (in-plane motion), the physics behind it comes from the smooth motion. The position results in xt + 1= x˙t⋅ t + xt and in velocity x˙t + 1= x˙t . For the state vector shown above, the dynamics in matrix notation is as follows:


This states "where" the state vector moves from one calculation step to the next within dt . This dynamic model is also called a "constant velocity" model because it assumes that the velocity remains constant during a filter's calculation step.


As an example, only the first line is written out, which calculates the position after a calculation step with the duration dt .

## *xt + 1= xt+ dt ⋅ x˙t*
This simply reflects physical relationships for the uniform motion. A higher form would be the Constant Acceleration model, which would be a 6D filter and still includes the accelerations in the state vector. In principle, other dynamics can be specified here (eg a  holonomic vehicle ).

#### Initial conditions / initialization
System state x
At the beginning you have to initialize with an initial state. In the 1 Dimensional case, μ0 , now in the multidimensional, was a vector.

If nothing is known, you can simply enter 0 here. If some boundary conditions are already known, they can be communicated to the filter. The choice of the following covariance matrix P controls how fast the filter converges to the correct (measured) values.

 ### Time Step between Filter Steps $dt$


```python
dt = 0.1
```

### Measurement Matrix $H$


The filter must also be told what is measured and how it relates to the state vector. In the example of the vehicle, what enters a tunnel, only the speed, not the position! The values ​​can be measured directly with the factor 1 (ie the velocity is measured directly in the correct unit), which is why in only 1.0 is set to the appropriate position.H


```python
A = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(A, A.shape)


# 
# 


H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(H, H.shape)
```

    [[1.  0.  0.1 0. ]
     [0.  1.  0.  0.1]
     [0.  0.  1.  0. ]
     [0.  0.  0.  1. ]] (4, 4)
    [[0. 0. 1. 0.]
     [0. 0. 0. 1.]] (2, 4)
    

### Measurement Noise Covariance $R$

Measurement noise covariance matrix R
As in the one-dimensional case the variance , a measurement uncertainty must also be stated here.σ0


This measurement uncertainty indicates how much one trusts the measured values ​​of the sensors. Since we measure only $\dot x$ and  $\dot y$ , this is a 2 × 2 matrix. If the sensor is very accurate, small values ​​should be used here. If the sensor is relatively inaccurate, large values ​​should be used here for  $\dot x$, $\dot y$


```python
ra = 10.0**2

R = np.matrix([[ra, 0.0],
              [0.0, ra]])
print(R, R.shape)
```

    [[100.   0.]
     [  0. 100.]] (2, 2)
    

#### Plot between -10 and 10 with .001 steps.


```python
xpdf = np.arange(-10, 10, 0.001)
plt.subplot(121)
plt.plot(xpdf, norm.pdf(xpdf,0,R[0,0]))
plt.title('$\dot x$')

plt.subplot(122)
plt.plot(xpdf, norm.pdf(xpdf,0,R[1,1]))
plt.title('$\dot y$')
plt.tight_layout()
```


![png](output_48_0.png)



```python
sv = 8.8

G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])

Q = G*G.T*sv**2


```


```python
Q 
```




    matrix([[0.001936, 0.001936, 0.03872 , 0.03872 ],
            [0.001936, 0.001936, 0.03872 , 0.03872 ],
            [0.03872 , 0.03872 , 0.7744  , 0.7744  ],
            [0.03872 , 0.03872 , 0.7744  , 0.7744  ]])



## Unit matrix $I$
Last but not least a unit matrix is ​​necessary.


```python
I = np.eye(4)
print(I, I.shape)
```

    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]] (4, 4)
    

## Measurements


```python
m = 200 # Measurements
vx= 30 # in X
vy= 10 # in Y

mx = np.array(vx+np.random.randn(m))
my = np.array(vy+np.random.randn(m))

measurements = np.vstack((mx,my))

print(measurements.shape)

print('Standard Deviation of Acceleration Measurements=%.2f' % np.std(mx))
print('You assumed %.2f in R.' % R[0,0])


fig = plt.figure(figsize=(16,5))

plt.step(range(m),mx, label='$\dot x$')
plt.step(range(m),my, label='$\dot y$')
plt.ylabel(r'Velocity $m/s$')
plt.title('Measurements')
plt.legend(loc='best',prop={'size':18})


```

    (2, 200)
    Standard Deviation of Acceleration Measurements=0.93
    You assumed 100.00 in R.
    




    <matplotlib.legend.Legend at 0x1c5733666d8>




![png](output_54_2.png)


# Preallocation for Plotting


```python
xt = []
yt = []
dxt= []
dyt= []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Rdx= []
Rdy= []
Kx = []
Ky = []
Kdx= []
Kdy= []

def savestates(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Rdx.append(float(R[0,0]))
    Rdy.append(float(R[1,1]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))  
```

## Kalman Filter Algorithm

## Filtering step Prediction / Predict
This part of the Kalman filter now dares to predict the state of the system in the future. In addition, under certain conditions (observability) a state can be calculated with it which can not be measured! That's amazing, but in our case exactly what we need. We can not measure the position of the vehicle because the GPS of the navigation device has no reception in a tunnel. By initializing the state vector with a position and measuring the velocity, however, the dynamics still be used to make an optimal prediction about the position.A

#### *xt+1=A⋅xt*

The covariance must also be recalculated. Uncertainty about the state of the system increases in the Predict step, as we have seen in the 1D case. In the multidimensional case, the measurement uncertainty Q is additively added, so the uncertainty P is getting bigger and bigger.PQP

#### *P=A⋅P⋅A‘+Q*

That's it. We have in the future ( ) expected. The Kalman filter has made a statement about the expected system state in the future.dt

Goes by now exactly time and the filter is in measuring / correcting and checking whether the prediction of the system state fits well with the new measurements. If so, the covariance chosen to be smaller by the filter (it is safer), if not, larger (something is wrong, the filter becomes more uncertain).dtP
Filter step Measure / Correct
The following mathematical calculations are not something that one must necessarily be able to derive. Rudolf E. Kalman thought about that in a few quiet minutes and it looks crazy, but works.

From the sensors come current measured values , with which an innovation  the state vector  with the measuring matrix  calculated.

#### *w=Z−(H⋅x)*
Then it is looked at with which variance (which in the multi-dimensional case is called covariance matrix) can be further calculated. For this, the uncertainty and the measurement matrix and the measurement uncertainty required.PHR
S=(H⋅P⋅H′+R)
This determines the so-called Kalman gain. It states whether the readings or system dynamics should be more familiar.

#### *K= P⋅ H'S*
The Kalman Gain will decrease if the readings match the predicted system state. If the measured values ​​say otherwise, the elements of matrix K become larger.

This information is now used to update the system state.

#### *x = x + ( K⋅ w )*
And also determined a new covariance for the upcoming Predict step.

#### *P= ( I- ( K⋅ H) ) ⋅ P*



```python
def kalman_filter(x,P):
    for n in range(len(measurements[0])):
 
        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x = A*x
    
        # Project the error covariance ahead
        P = A*P*A.T + Q
    
    
        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)

    
        # Update the estimate via z
        Z = measurements[:,n].reshape(2,1)
        y = Z - (H*x)                            # Innovation or Residual
        x = x + (K*y)
    
        # Update the error covariance
        P = (I - (K*H))*P
    
    
    
        # Save states (for Plotting)
        savestates(x, Z, P, R, K)
        print("x:", x)
        print("P:", P)

```

### State Estimate $x$


```python
def plot_x():
    fig = plt.figure(figsize=(16,9))
    plt.step(range(len(measurements[0])),dxt, label='$\dot x$')
    plt.step(range(len(measurements[0])),dyt, label='$\dot y$')

    plt.axhline(vx, color='#999999', label='$\dot x_{real}$')
    plt.axhline(vy, color='#999999', label='$\dot y_{real}$')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best',prop={'size':22})
    plt.ylim([0, 50])
    plt.ylabel('Velocity')



plot_x()
```


![png](output_61_0.png)



```python
def plot_K():
    fig = plt.figure(figsize=(16,9))
    plt.plot(range(len(measurements[0])),Kx, label='Kalman Gain for $x$')
    plt.plot(range(len(measurements[0])),Ky, label='Kalman Gain for $y$')
    plt.plot(range(len(measurements[0])),Kdx, label='Kalman Gain for $\dot x$')
    plt.plot(range(len(measurements[0])),Kdy, label='Kalman Gain for $\dot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
    plt.legend(loc='best',prop={'size':22})



plot_K()
```


![png](output_62_0.png)



```python
def plot_P():
    fig = plt.figure(figsize=(16,9))
    plt.plot(range(len(measurements[0])),Px, label='$x$')
    plt.plot(range(len(measurements[0])),Py, label='$y$')
    plt.plot(range(len(measurements[0])),Pdx, label='$\dot x$')
    plt.plot(range(len(measurements[0])),Pdy, label='$\dot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best',prop={'size':22})


plot_P()
```


![png](output_63_0.png)



```python
time_stamp = []
for i in range(200):
    i+=1
    time_stamp.append(i)
print (time_stamp)
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    


```python
xt
```




    [2.6060861435401446,
     5.500666914802907,
     8.723887927797294,
     11.756217983178695,
     14.911657090891586,
     17.860866545165578,
     20.70519293734818,
     23.721351792414207,
     26.658424127462116,
     29.671206159357116,
     32.59824664338847,
     35.46022029660513,
     38.63342496131479,
     41.580354204343585,
     44.59723714872179,
     47.773835174001476,
     50.8573675351977,
     53.75193478927431,
     56.73783298480521,
     59.709209087486975,
     62.72290725174733,
     65.65562271505694,
     68.59624298585774,
     71.51232097063482,
     74.46170615513773,
     77.40178176311676,
     80.40470908293842,
     83.31067567548591,
     86.46426900888358,
     89.4374250656544,
     92.48846518654915,
     95.51917309384376,
     98.56941937080616,
     101.64680167362417,
     104.75448998567099,
     107.73037560307115,
     110.75129156670603,
     113.61709464238005,
     116.67845893083563,
     119.7223742106172,
     122.75005517658944,
     125.68740424327788,
     128.68710356507054,
     131.87480035276053,
     134.71050212569045,
     137.79605303095883,
     140.68597190992952,
     143.7879426126336,
     146.62885897446913,
     149.54012312127296,
     152.67682632336258,
     155.65407840905968,
     158.6743788061528,
     161.7841328392158,
     164.795748537179,
     167.86578518099884,
     170.97897771431437,
     174.0627520582977,
     177.00698139604805,
     179.85545522461894,
     182.84724110792354,
     185.85641219458572,
     188.83407941681304,
     191.9366510533077,
     195.1387049502023,
     198.260727578883,
     201.1186972901865,
     204.2798297870288,
     207.2722812719448,
     210.31231856274613,
     213.45918854918705,
     216.56176384694785,
     219.64406088048796,
     222.6195091690789,
     225.72775748622155,
     228.64844327421662,
     231.73903176676328,
     234.7356131136471,
     237.71544881886493,
     240.74599801880603,
     243.8802021807265,
     246.8228070906887,
     249.86216246038452,
     252.93535702540194,
     255.91932229264773,
     258.83166490119305,
     261.9125942299241,
     264.9146361849297,
     267.9060713535007,
     270.8833154059299,
     273.8647589182071,
     276.8359224510442,
     279.73720495580955,
     282.72393197507665,
     285.70636147054,
     288.70644786245504,
     291.7802028394281,
     294.84685529067025,
     297.87466760531726,
     300.73954225548226,
     303.7726427902319,
     306.77543516355183,
     309.6579526673112,
     312.6830674012583,
     315.81372730638395,
     318.91353219835116,
     321.9632127094075,
     325.088690229924,
     327.9638946779017,
     330.8677467326701,
     333.79281135009705,
     336.9072005276412,
     339.8821850295275,
     342.86260037036635,
     345.81205154227024,
     348.7566643254535,
     351.83158552005983,
     354.79773170213053,
     357.8885525379875,
     361.04868102868795,
     364.1568484615835,
     367.0643261245263,
     370.10134981683893,
     372.98381258161277,
     375.99654851324107,
     378.967610173132,
     382.1035834416404,
     385.17691052266343,
     388.32537637968767,
     391.3002839632256,
     394.40340918702327,
     397.3748461508699,
     400.28732772036204,
     403.26752383483745,
     406.18805066832294,
     409.2353634083848,
     412.15950346059606,
     415.1385069809749,
     418.1674550862901,
     421.0906654607073,
     424.18023075255905,
     427.0673126577098,
     430.13465536191416,
     433.1024069894285,
     436.12583695726863,
     439.06951363495443,
     442.03137128483223,
     444.98675031267607,
     447.9096423337697,
     450.78744273860633,
     453.73385041704137,
     456.8774466283733,
     459.775114204115,
     463.00066572872396,
     465.89762762476505,
     468.87609971294916,
     471.7883924954053,
     474.77948420367704,
     477.64382661222857,
     480.6776474166985,
     483.6295024760669,
     486.5214720084545,
     489.5280480449368,
     492.5926619731092,
     495.728028468867,
     498.78846965179935,
     501.77435287903637,
     504.86674877089155,
     507.8025339000299,
     510.7190054213957,
     513.6935330456537,
     516.7586196504576,
     519.6408202280932,
     522.9018650837681,
     525.9020249252734,
     528.9787609223398,
     531.9611917542642,
     534.9569702357152,
     537.9255904986035,
     540.8816455550889,
     543.7112200981952,
     546.7909838503705,
     549.617930617988,
     552.5493382075385,
     555.4587798141406,
     558.6626576471715,
     561.7099650686636,
     564.7720662677535,
     567.8232971371274,
     570.7527853752883,
     573.7568712558573,
     576.8034659313295,
     579.7187502761616,
     582.7599014801325,
     585.7059201697973,
     588.5699586684683,
     591.714746967836,
     594.7644713521621,
     597.7502102241475,
     600.8938991007774]




```python
def plot_xy():
    fig = plt.figure(figsize=(16,16))
    
   
    plt.scatter(xt,yt, s=20, label='State', c='k')
    
    plt.scatter(xt[0],yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1],yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
plot_xy()
```


![png](output_66_0.png)



```python
def plot_x():
    fig = plt.figure(figsize=(16,16))
    
    plt.scatter(time_stamp,measurements[0], s=20, label='State', c='k')
    
    plt.scatter(time_stamp[0],measurements[0][0], s=100, label='Start', c='g')
    plt.scatter(time_stamp[-1],measurements[0][-1], s=100, label='Goal', c='r')
   
    plt.scatter(time_stamp,dxt, s=30, label='State', c='y')
    
    plt.scatter(time_stamp[0],dxt[0], s=100, label='Start', c='g')
    plt.scatter(time_stamp[-1],dxt[-1], s=100, label='Goal', c='r')

    plt.xlabel('Time Step')
    plt.ylabel('dX')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
plot_x()
```


![png](output_67_0.png)



```python
def plot_y():
    fig = plt.figure(figsize=(16,16))
    
    plt.scatter(measurements[1],time_stamp, s=20, label='State', c='k')
    
    plt.scatter(measurements[1][0],time_stamp[0], s=100, label='Start', c='g')
    plt.scatter(measurements[1][-1],time_stamp[-1], s=100, label='Goal', c='r')
   
    plt.scatter(dyt,time_stamp, s=20, label='State', c='y')
    plt.scatter(dyt[0],time_stamp[0], s=100, label='Start', c='g')
    plt.scatter(dyt[-1],time_stamp[-1], s=100, label='Goal', c='r')

    plt.xlabel('dY - Velocity')
    plt.ylabel('Time Step')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
plot_y()
```


![png](output_68_0.png)


### Kalman Filter - Udacity Assignment


```python
xt = []
yt = []
dxt= []
dyt= []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Rdx= []
Rdy= []
Kx = []
Ky = []
Kdx= []
Kdy= []
```


```python
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

def filter(x, P):
    plt.scatter([x[0]], [x[1]], s=100)
    plt.title('Initial Location')

    for n in range(len(measurements)):

        # prediction
        x = (F * x) + u
        P = (F * P * F.transpose()) + Q
        

        # measurement update
        Z = np.matrix(measurements[n])
        
        y = Z.transpose() - (H * x)
       
        S = H * P * H.transpose() + R
        K = P * H.transpose() * inv(S)
        x = x + (K * y)
        P = (I - (K * H)) * P
               
        xt.append(float(x[0]))
        yt.append(float(x[1]))
        dxt.append(float(x[2]))
        dyt.append(float(x[3]))
        #Zx.append(float(Z[0]))
        #Zy.append(float(Z[1]))
        Px.append(float(P[0,0]))
        Py.append(float(P[1,1]))
        Pdx.append(float(P[2,2]))
        Pdy.append(float(P[3,3]))
        Rdx.append(float(R[0,0]))
        Rdy.append(float(R[1,1]))
        Kx.append(float(K[0,0]))
        Ky.append(float(K[1,0]))
        Kdx.append(float(K[2,0]))
        Kdy.append(float(K[3,0]))
        #print('X:',x)
        #print(P)


dt = 0.1
u = np.matrix([[0.], [0.], [0.], [0.]])
measurements = np.matrix([[5.0, 10.0], [6.0, 8.0], [7.0, 6.0], [8.0, 4.0], [9.0, 2.0], [10.0, 0.0]])
#measurements = np.matrix([[1., 4.], [6., 0.], [11., -4.], [16., -8.]])
#measurements = np.matrix([[1., 17.], [1., 15.], [1., 13.], [1., 11.]])


x = np.matrix([[4.], [12.], [0.0], [0.0]])# initial state (location and velocity)
#x = np.matrix([[-4.], [8.], [0.0], [0.0]])
#x = np.matrix([[1.], [19.], [0.0], [0.0]])

P = np.matrix([[0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1000, 0.0],
              [0.0, 0.0, 0.0, 1000]]) # initial uncertainty: 0 for positions x and y, 1000 for the two velocities

F = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]) # next state function: generalize the 2d version to 4d


H = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]) # measurement function: reflect the fact that we observe x and y but not the two velocities

R = np.matrix([[1.0, 0.0], [0.0, 1.0]]) # measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal

I = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # 4d identity matrix

sv = 1.0

G = np.matrix([[dt**2],
               [dt**2],
               [dt],
               [dt]])


Q = G*G.T*sv**2





```


```python
filter(x,P)
```


![png](output_72_0.png)


#### Plot the Estimated and Updated Points/


```python
fig = plt.figure(figsize=(16,16))
plt.scatter(xt,yt, s=20, label='State', c='k')
plt.scatter(xt[0],yt[0], s=100, label='Start', c='g')
plt.scatter(xt[-1],yt[-1], s=100, label='Goal', c='r')


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.legend(loc='best')
plt.axis('equal')
```




    (4.620378169645521,
     10.282291361818729,
     -0.5664510646025152,
     10.761038093342572)




![png](output_74_1.png)



```python
()
```




    ()


