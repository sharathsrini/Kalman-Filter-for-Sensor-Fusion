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




    array([ 9.54839223, 19.89699091, 29.06691149, 42.33270256, 49.16466159])




```python
distances
```




    array([9.23285848, 8.91488919, 8.46738816, 9.15677889, 9.32091845])




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

    After correction:  mean= 19.24	var= 7.29
    After correction:  mean= 23.28	var= 7.08
    After correction:  mean= 30.17	var= 7.05
    After correction:  mean= 41.09	var= 7.04
    After correction:  mean= 49.68	var= 7.04
    


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
    




    Text(0.5,1,'Initial Location')




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
vx= 20 # in X
vy= 10 # in Y

mx = np.array(vx+5*np.random.randn(m))
my = np.array(vy+5*np.random.randn(m))

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
    Standard Deviation of Acceleration Measurements=4.98
    You assumed 100.00 in R.
    




    <matplotlib.legend.Legend at 0x28ad1050c18>




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


```python
kalman_filter(x,P)
```

    x: [[ 1.6176229 ]
     [ 1.35433895]
     [16.18772752]
     [13.554888  ]]
    P: [[ 1.00091039e+03  1.29417780e-03  9.08803314e+00 -2.87595066e-03]
     [ 1.29417780e-03  1.00091039e+03 -2.87595066e-03  9.08803314e+00]
     [ 9.08803314e+00 -2.87595066e-03  9.09154819e+01  6.39100147e-03]
     [-2.87595066e-03  9.08803314e+00  6.39100147e-03  9.09154819e+01]]
    x: [[ 3.32772989]
     [ 2.35236628]
     [16.63567529]
     [11.75885725]]
    P: [[ 1.00190760e+03  2.83682229e-03  9.50412970e+00 -1.96798285e-02]
     [ 2.83682229e-03  1.00190760e+03 -1.96798285e-02  9.50412970e+00]
     [ 9.50412970e+00 -1.96798285e-02  4.78315401e+01  2.12492509e-01]
     [-1.96798285e-02  9.50412970e+00  2.12492509e-01  4.78315401e+01]]
    x: [[ 4.84578698]
     [ 3.61873374]
     [16.14737473]
     [12.05719726]]
    P: [[ 1.00290759e+03  4.36555678e-03  9.64050829e+00 -3.69110627e-02]
     [ 4.36555678e-03  1.00290759e+03 -3.69110627e-02  9.64050829e+00]
     [ 9.64050829e+00 -3.69110627e-02  3.27049710e+01  4.46906496e-01]
     [-3.69110627e-02  9.64050829e+00  4.46906496e-01  3.27049710e+01]]
    x: [[ 6.51713806]
     [ 4.6851181 ]
     [16.28187498]
     [11.70182509]]
    P: [[ 1.00390830e+03  5.86151856e-03  9.70216148e+00 -5.39360809e-02]
     [ 5.86151856e-03  1.00390830e+03 -5.39360809e-02  9.70216148e+00]
     [ 9.70216148e+00 -5.39360809e-02  2.50757838e+01  6.85539880e-01]
     [-5.39360809e-02  9.70216148e+00  6.85539880e-01  2.50757838e+01]]
    x: [[ 7.9695235 ]
     [ 6.01694363]
     [15.92871329]
     [12.02355356]]
    P: [[ 1.00490927e+03  7.31310944e-03  9.73339029e+00 -7.05312752e-02]
     [ 7.31310944e-03  1.00490927e+03 -7.05312752e-02  9.73339029e+00]
     [ 9.73339029e+00 -7.05312752e-02  2.05297472e+01  9.21904028e-01]
     [-7.05312752e-02  9.73339029e+00  9.21904028e-01  2.05297472e+01]]
    x: [[10.3917381 ]
     [ 7.55170968]
     [17.4669223 ]
     [12.73354161]]
    P: [[ 1.00591035e+03  8.71114345e-03  9.74950455e+00 -8.65610231e-02]
     [ 8.71114345e-03  1.00591035e+03 -8.65610231e-02  9.74950455e+00]
     [ 9.74950455e+00 -8.65610231e-02  1.75464639e+01  1.15302130e+00]
     [-8.65610231e-02  9.74950455e+00  1.15302130e+00  1.75464639e+01]]
    x: [[12.11160797]
     [ 8.67707212]
     [17.40103497]
     [12.49455519]]
    P: [[ 1.00691146e+03  1.00482045e-02  9.75722720e+00 -1.01927728e-01]
     [ 1.00482045e-02  1.00691146e+03 -1.01927728e-01  9.75722720e+00]
     [ 9.75722720e+00 -1.01927728e-01  1.54616190e+01  1.37711195e+00]
     [-1.01927728e-01  9.75722720e+00  1.37711195e+00  1.54616190e+01]]
    x: [[13.60235521]
     [10.02439379]
     [17.06202642]
     [12.58957464]]
    P: [[ 1.00791255e+03  1.13185217e-02  9.75998403e+00 -1.16559184e-01]
     [ 1.13185217e-02  1.00791255e+03 -1.16559184e-01  9.75998403e+00]
     [ 9.75998403e+00 -1.16559184e-01  1.39386620e+01  1.59298296e+00]
     [-1.16559184e-01  9.75998403e+00  1.59298296e+00  1.39386620e+01]]
    x: [[16.11086938]
     [10.25090254]
     [17.90700578]
     [11.39593152]]
    P: [[ 1.00891362e+03  1.25179118e-02  9.75970582e+00 -1.30404069e-01]
     [ 1.25179118e-02  1.00891362e+03 -1.30404069e-01  9.75970582e+00]
     [ 9.75970582e+00 -1.30404069e-01  1.27888256e+01  1.79981464e+00]
     [-1.30404069e-01  9.75970582e+00  1.79981464e+00  1.27888256e+01]]
    x: [[18.36208314]
     [10.96472802]
     [18.37525472]
     [10.9778996 ]]
    P: [[ 1.00991463e+03  1.36436999e-02  9.75756061e+00 -1.43429493e-01]
     [ 1.36436999e-02  1.00991463e+03 -1.43429493e-01  9.75756061e+00]
     [ 9.75756061e+00 -1.43429493e-01  1.18980566e+01  1.99706649e+00]
     [-1.43429493e-01  9.75756061e+00  1.99706649e+00  1.18980566e+01]]
    x: [[20.04125567]
     [12.01384488]
     [18.18113213]
     [10.88348596]]
    P: [[ 1.01091560e+03  1.46946046e-02  9.75429091e+00 -1.55618998e-01]
     [ 1.46946046e-02  1.01091560e+03 -1.55618998e-01  9.75429091e+00]
     [ 9.75429091e+00 -1.55618998e-01  1.11934331e+01  2.18442405e+00]
     [-1.55618998e-01  9.75429091e+00  2.18442405e+00  1.11934331e+01]]
    x: [[21.26361734]
     [12.96883481]
     [17.49440273]
     [10.58208395]]
    P: [[ 1.01191650e+03  1.56705920e-02  9.75038481e+00 -1.66970565e-01]
     [ 1.56705920e-02  1.01191650e+03 -1.66970565e-01  9.75038481e+00]
     [ 9.75038481e+00 -1.66970565e-01  1.06262240e+01  2.36176115e+00]
     [-1.66970565e-01  9.75038481e+00  2.36176115e+00  1.06262240e+01]]
    x: [[22.73031123]
     [14.22287966]
     [17.25269413]
     [10.708516  ]]
    P: [[ 1.01291734e+03  1.65727076e-02  9.74616957e+00 -1.77494548e-01]
     [ 1.65727076e-02  1.01291734e+03 -1.77494548e-01  9.74616957e+00]
     [ 9.74616957e+00 -1.77494548e-01  1.01626963e+01  2.52910847e+00]
     [-1.77494548e-01  9.74616957e+00  2.52910847e+00  1.01626963e+01]]
    x: [[24.32557545]
     [14.61426936]
     [16.92089436]
     [ 9.98424716]]
    P: [[ 1.01391811e+03  1.74028970e-02  9.74186644e+00 -1.87211574e-01]
     [ 1.74028970e-02  1.01391811e+03 -1.87211574e-01  9.74186644e+00]
     [ 9.74186644e+00 -1.87211574e-01  9.77882397e+00  2.68662539e+00]
     [-1.87211574e-01  9.74186644e+00  2.68662539e+00  9.77882397e+00]]
    x: [[26.8075284 ]
     [15.66861708]
     [17.71032728]
     [10.28438641]]
    P: [[ 1.01491883e+03  1.81638278e-02  9.73762435e+00 -1.96150486e-01]
     [ 1.81638278e-02  1.01491883e+03 -1.96150486e-01  9.73762435e+00]
     [ 9.73762435e+00 -1.96150486e-01  9.45709068e+00  2.83457412e+00]
     [-1.96150486e-01  9.73762435e+00  2.83457412e+00  9.45709068e+00]]
    x: [[29.60159743]
     [17.12460816]
     [18.82177891]
     [11.02366061]]
    P: [[ 1.01591948e+03  1.88587181e-02  9.73354180e+00 -2.04346396e-01]
     [ 1.88587181e-02  1.01591948e+03 -2.04346396e-01  9.73354180e+00]
     [ 9.73354180e+00 -2.04346396e-01  9.18447635e+00  2.97329622e+00]
     [-2.04346396e-01  9.73354180e+00  2.97329622e+00  9.18447635e+00]]
    x: [[31.86626885]
     [17.2764775 ]
     [18.85414718]
     [10.27191698]]
    P: [[ 1.01692008e+03  1.94911786e-02  9.72968156e+00 -2.11838907e-01]
     [ 1.94911786e-02  1.01692008e+03 -2.11838907e-01  9.72968156e+00]
     [ 9.72968156e+00 -2.11838907e-01  8.95114472e+00  3.10319151e+00]
     [-2.11838907e-01  9.72968156e+00  3.10319151e+00  8.95114472e+00]]
    x: [[33.55600161]
     [18.6370368 ]
     [18.79389417]
     [10.50558039]]
    P: [[ 1.01792062e+03  2.00650737e-02  9.72608084e+00 -2.18670540e-01]
     [ 2.00650737e-02  1.01792062e+03 -2.18670540e-01  9.72608084e+00]
     [ 9.72608084e+00 -2.18670540e-01  8.74956152e+00  3.22469965e+00]
     [-2.18670540e-01  9.72608084e+00  3.22469965e+00  8.74956152e+00]]
    x: [[35.91237203]
     [19.54321741]
     [19.16598417]
     [10.55063963]]
    P: [[ 1.01892111e+03  2.05844008e-02  9.72275860e+00 -2.24885383e-01]
     [ 2.05844008e-02  1.01892111e+03 -2.24885383e-01  9.72275860e+00]
     [ 9.72275860e+00 -2.24885383e-01  8.57388655e+00  3.33828445e+00]
     [-2.24885383e-01  9.72275860e+00  3.33828445e+00  8.57388655e+00]]
    x: [[37.21837287]
     [19.38117066]
     [18.17505642]
     [ 9.25645532]]
    P: [[ 1.01992155e+03  2.10531903e-02  9.71972079e+00 -2.30527968e-01]
     [ 2.10531903e-02  1.01992155e+03 -2.30527968e-01  9.71972079e+00]
     [ 9.71972079e+00 -2.30527968e-01  8.41954518e+00  3.44442080e+00]
     [-2.30527968e-01  9.71972079e+00  3.44442080e+00  8.41954518e+00]]
    x: [[39.10508773]
     [21.07523705]
     [18.53098541]
     [ 9.94534223]]
    P: [[ 1.02092195e+03  2.14754251e-02  9.71696427e+00 -2.35642368e-01]
     [ 2.14754251e-02  1.02092195e+03 -2.35642368e-01  9.71696427e+00]
     [ 9.71696427e+00 -2.35642368e-01  8.28292054e+00  3.54358405e+00]
     [-2.35642368e-01  9.71696427e+00  3.54358405e+00  8.28292054e+00]]
    x: [[41.15657573]
     [21.48731321]
     [18.4693279 ]
     [ 9.52875402]]
    P: [[ 1.02192231e+03  2.18549774e-02  9.71447965e+00 -2.40271486e-01]
     [ 2.18549774e-02  1.02192231e+03 -2.40271486e-01  9.71447965e+00]
     [ 9.71447965e+00 -2.40271486e-01  8.16112863e+00  3.63624175e+00]
     [-2.40271486e-01  9.71447965e+00  3.63624175e+00  8.16112863e+00]]
    x: [[42.94416078]
     [22.29468217]
     [18.36067381]
     [ 9.38263963]]
    P: [[ 1.02292263e+03  2.21955622e-02  9.71225342e+00 -2.44456532e-01]
     [ 2.21955622e-02  1.02292263e+03 -2.44456532e-01  9.71225342e+00]
     [ 9.71225342e+00 -2.44456532e-01  8.05185166e+00  3.72284733e+00]
     [-2.44456532e-01  9.71225342e+00  3.72284733e+00  8.05185166e+00]]
    x: [[45.12907302]
     [24.26646397]
     [19.07686096]
     [10.38410719]]
    P: [[ 1.02392292e+03  2.25007040e-02  9.71026958e+00 -2.48236647e-01]
     [ 2.25007040e-02  1.02392292e+03 -2.48236647e-01  9.71026958e+00]
     [ 9.71026958e+00 -2.48236647e-01  7.95321316e+00  3.80383556e+00]
     [-2.48236647e-01  9.71026958e+00  3.80383556e+00  7.95321316e+00]]
    x: [[47.60275364]
     [24.97797592]
     [19.40389129]
     [10.3539802 ]]
    P: [[ 1.02492317e+03  2.27737150e-02  9.70851070e+00 -2.51648659e-01]
     [ 2.27737150e-02  1.02492317e+03 -2.51648659e-01  9.70851070e+00]
     [ 9.70851070e+00 -2.51648659e-01  7.86368319e+00  3.87961945e+00]
     [-2.51648659e-01  9.70851070e+00  3.87961945e+00  7.86368319e+00]]
    x: [[49.67841478]
     [25.15509853]
     [19.14624368]
     [ 9.71419897]]
    P: [[ 1.02592340e+03  2.30176834e-02  9.70695888e+00 -2.54726943e-01]
     [ 2.30176834e-02  1.02592340e+03 -2.54726943e-01  9.70695888e+00]
     [ 9.70695888e+00 -2.54726943e-01  7.78200590e+00  3.95058828e+00]
     [-2.54726943e-01  9.70695888e+00  3.95058828e+00  7.78200590e+00]]
    x: [[51.52377768]
     [26.24707941]
     [19.14292047]
     [ 9.78118036]]
    P: [[ 1.02692360e+03  2.32354686e-02  9.70559626e+00 -2.57503368e-01]
     [ 2.32354686e-02  1.02692360e+03 -2.57503368e-01  9.70559626e+00]
     [ 9.70559626e+00 -2.57503368e-01  7.70714359e+00  4.01710669e+00]
     [-2.57503368e-01  9.70559626e+00  4.01710669e+00  7.70714359e+00]]
    x: [[52.94316534]
     [27.49265497]
     [18.86568675]
     [ 9.77621876]]
    P: [[ 1.02792379e+03  2.34297025e-02  9.70440551e+00 -2.60007305e-01]
     [ 2.34297025e-02  1.02792379e+03 -2.60007305e-01  9.70440551e+00]
     [ 9.70440551e+00 -2.60007305e-01  7.63823330e+00  4.07951444e+00]
     [-2.60007305e-01  9.70440551e+00  4.07951444e+00  7.63823330e+00]]
    x: [[54.56207111]
     [29.0953903 ]
     [18.93348722]
     [10.15187315]]
    P: [[ 1.02892395e+03  2.36027957e-02  9.70337005e+00 -2.62265690e-01]
     [ 2.36027957e-02  1.02892395e+03 -2.62265690e-01  9.70337005e+00]
     [ 9.70337005e+00 -2.62265690e-01  7.57455282e+00  4.13812670e+00]
     [-2.62265690e-01  9.70337005e+00  4.13812670e+00  7.57455282e+00]]
    x: [[56.90132285]
     [30.0685309 ]
     [19.26531855]
     [10.32105457]]
    P: [[ 1.02992409e+03  2.37569455e-02  9.70247429e+00 -2.64303118e-01]
     [ 2.37569455e-02  1.02992409e+03 -2.64303118e-01  9.70247429e+00]
     [ 9.70247429e+00 -2.64303118e-01  7.51549399e+00  4.19323485e+00]
     [-2.64303118e-01  9.70247429e+00  4.19323485e+00  7.51549399e+00]]
    x: [[58.69343381]
     [31.74606914]
     [19.45651343]
     [10.76381515]]
    P: [[ 1.03092422e+03  2.38941478e-02  9.70170370e+00 -2.66141960e-01]
     [ 2.38941478e-02  1.03092422e+03 -2.66141960e-01  9.70170370e+00]
     [ 9.70170370e+00 -2.66141960e-01  7.46054156e+00  4.24510748e+00]
     [-2.66141960e-01  9.70170370e+00  4.24510748e+00  7.46054156e+00]]
    x: [[60.59073683]
     [33.35688817]
     [19.66698373]
     [11.15640602]]
    P: [[ 1.03192433e+03  2.40162092e-02  9.70104485e+00 -2.67802504e-01]
     [ 2.40162092e-02  1.03192433e+03 -2.67802504e-01  9.70104485e+00]
     [ 9.70104485e+00 -2.67802504e-01  7.40925640e+00  4.29399160e+00]
     [-2.67802504e-01  9.70104485e+00  4.29399160e+00  7.40925640e+00]]
    x: [[63.07319193]
     [33.98783005]
     [19.83783491]
     [11.02408888]]
    P: [[ 1.03292443e+03  2.41247600e-02  9.70048543e+00 -2.69303093e-01]
     [ 2.41247600e-02  1.03292443e+03 -2.69303093e-01  9.70048543e+00]
     [ 9.70048543e+00 -2.69303093e-01  7.36126203e+00  4.34011400e+00]
     [-2.69303093e-01  9.70048543e+00  4.34011400e+00  7.36126203e+00]]
    x: [[65.38823751]
     [34.77092988]
     [19.94092278]
     [10.93583229]]
    P: [[ 1.03392451e+03  2.42212679e-02  9.70001421e+00 -2.70660273e-01]
     [ 2.42212679e-02  1.03392451e+03 -2.70660273e-01  9.70001421e+00]
     [ 9.70001421e+00 -2.70660273e-01  7.31623391e+00  4.38368259e+00]
     [-2.70660273e-01  9.70001421e+00  4.38368259e+00  7.31623391e+00]]
    x: [[66.95467603]
     [34.84824176]
     [19.12913489]
     [ 9.95586795]]
    P: [[ 1.03492459e+03  2.43070510e-02  9.69962103e+00 -2.71888939e-01]
     [ 2.43070510e-02  1.03492459e+03 -2.71888939e-01  9.69962103e+00]
     [ 9.69962103e+00 -2.71888939e-01  7.27389068e+00  4.42488783e+00]
     [-2.71888939e-01  9.69962103e+00  4.42488783e+00  7.27389068e+00]]
    x: [[68.16785711]
     [36.5708013 ]
     [18.94788382]
     [10.17092387]]
    P: [[ 1.03592466e+03  2.43832911e-02  9.69929669e+00 -2.73002476e-01]
     [ 2.43832911e-02  1.03592466e+03 -2.73002476e-01  9.69929669e+00]
     [ 9.69929669e+00 -2.73002476e-01  7.23398714e+00  4.46390404e+00]
     [-2.73002476e-01  9.69929669e+00  4.46390404e+00  7.23398714e+00]]
    x: [[70.06530866]
     [37.0934235 ]
     [18.70987819]
     [ 9.79855788]]
    P: [[ 1.03692472e+03  2.44510455e-02  9.69903293e+00 -2.74012892e-01]
     [ 2.44510455e-02  1.03692472e+03 -2.74012892e-01  9.69903293e+00]
     [ 9.69903293e+00 -2.74012892e-01  7.19630855e+00  4.50089076e+00]
     [-2.74012892e-01  9.69903293e+00  4.50089076e+00  7.19630855e+00]]
    x: [[72.17667663]
     [37.8796562 ]
     [18.79600064]
     [ 9.77046895]]
    P: [[ 1.03792477e+03  2.45112590e-02  9.69882233e+00 -2.74930950e-01]
     [ 2.45112590e-02  1.03792477e+03 -2.74930950e-01  9.69882233e+00]
     [ 9.69882233e+00 -2.74930950e-01  7.16066587e+00  4.53599396e+00]
     [-2.74930950e-01  9.69882233e+00  4.53599396e+00  7.16066587e+00]]
    x: [[73.21784653]
     [39.59467385]
     [18.53154994]
     [ 9.91022361]]
    P: [[ 1.03892482e+03  2.45647746e-02  9.69865827e+00 -2.75766281e-01]
     [ 2.45647746e-02  1.03892482e+03 -2.75766281e-01  9.69865827e+00]
     [ 9.69865827e+00 -2.75766281e-01  7.12689196e+00  4.56934721e+00]
     [-2.75766281e-01  9.69865827e+00  4.56934721e+00  7.12689196e+00]]
    x: [[75.88680173]
     [40.80839062]
     [19.25025043]
     [10.48064765]]
    P: [[ 1.03992486e+03  2.46123439e-02  9.69853484e+00 -2.76527500e-01]
     [ 2.46123439e-02  1.03992486e+03 -2.76527500e-01  9.69853484e+00]
     [ 9.69853484e+00 -2.76527500e-01  7.09483837e+00  4.60107279e+00]
     [-2.76527500e-01  9.69853484e+00  4.60107279e+00  7.09483837e+00]]
    x: [[78.65090367]
     [41.60271155]
     [19.74684205]
     [10.71069764]]
    P: [[ 1.04092490e+03  2.46546358e-02  9.69844679e+00 -2.77222305e-01]
     [ 2.46546358e-02  1.04092490e+03 -2.77222305e-01  9.69844679e+00]
     [ 9.69844679e+00 -2.77222305e-01  7.06437270e+00  4.63128268e+00]
     [-2.77222305e-01  9.69844679e+00  4.63128268e+00  7.06437270e+00]]
    x: [[81.5753713 ]
     [42.14637754]
     [20.18488683]
     [10.79703117]]
    P: [[ 1.04192493e+03  2.46922456e-02  9.69838946e+00 -2.77857573e-01]
     [ 2.46922456e-02  1.04192493e+03 -2.77857573e-01  9.69838946e+00]
     [ 9.69838946e+00 -2.77857573e-01  7.03537643e+00  4.66007951e+00]
     [-2.77857573e-01  9.69838946e+00  4.66007951e+00  7.03537643e+00]]
    x: [[83.22539144]
     [43.21392635]
     [19.9071738 ]
     [10.60218192]]
    P: [[ 1.04292496e+03  2.47257025e-02  9.69835870e+00 -2.78439441e-01]
     [ 2.47257025e-02  1.04292496e+03 -2.78439441e-01  9.69835870e+00]
     [ 9.69835870e+00 -2.78439441e-01  7.00774302e+00  4.68755740e+00]
     [-2.78439441e-01  9.69835870e+00  4.68755740e+00  7.00774302e+00]]
    x: [[85.07731001]
     [44.23157479]
     [19.7836443 ]
     [10.50052266]]
    P: [[ 1.04392498e+03  2.47554764e-02  9.69835088e+00 -2.78973386e-01]
     [ 2.47554764e-02  1.04392498e+03 -2.78973386e-01  9.69835088e+00]
     [ 9.69835088e+00 -2.78973386e-01  6.98137642e+00  4.71380273e+00]
     [-2.78973386e-01  9.69835088e+00  4.71380273e+00  6.98137642e+00]]
    x: [[87.27176637]
     [44.44618517]
     [19.51596986]
     [ 9.99917404]]
    P: [[ 1.04492500e+03  2.47819846e-02  9.69836275e+00 -2.79464297e-01]
     [ 2.47819846e-02  1.04492500e+03 -2.79464297e-01  9.69836275e+00]
     [ 9.69836275e+00 -2.79464297e-01  6.95618974e+00  4.73889484e+00]
     [-2.79464297e-01  9.69836275e+00  4.73889484e+00  6.95618974e+00]]
    x: [[88.95039164]
     [45.44270973]
     [19.31508555]
     [ 9.85689383]]
    P: [[ 1.04592502e+03  2.48055975e-02  9.69839149e+00 -2.79916532e-01]
     [ 2.48055975e-02  1.04592502e+03 -2.79916532e-01  9.69839149e+00]
     [ 9.69839149e+00 -2.79916532e-01  6.93210412e+00  4.76290672e+00]
     [-2.79916532e-01  9.69839149e+00  4.76290672e+00  6.93210412e+00]]
    x: [[90.47809884]
     [46.57727984]
     [19.09801537]
     [ 9.75741559]]
    P: [[ 1.04692504e+03  2.48266435e-02  9.69843460e+00 -2.80333981e-01]
     [ 2.48266435e-02  1.04692504e+03 -2.80333981e-01  9.69843460e+00]
     [ 9.69843460e+00 -2.80333981e-01  6.90904780e+00  4.78590555e+00]
     [-2.80333981e-01  9.69843460e+00  4.78590555e+00  6.90904780e+00]]
    x: [[91.79637577]
     [47.26013129]
     [18.51777818]
     [ 9.23939391]]
    P: [[ 1.04792505e+03  2.48454139e-02  9.69848987e+00 -2.80720113e-01]
     [ 2.48454139e-02  1.04792505e+03 -2.80720113e-01  9.69848987e+00]
     [ 9.69848987e+00 -2.80720113e-01  6.88695529e+00  4.80795322e+00]
     [-2.80720113e-01  9.69848987e+00  4.80795322e+00  6.88695529e+00]]
    x: [[93.88921564]
     [47.21024528]
     [18.18676379]
     [ 8.66044331]]
    P: [[ 1.04892507e+03  2.48621667e-02  9.69855538e+00 -2.81078023e-01]
     [ 2.48621667e-02  1.04892507e+03 -2.81078023e-01  9.69855538e+00]
     [ 9.69855538e+00 -2.81078023e-01  6.86576671e+00  4.82910683e+00]
     [-2.81078023e-01  9.69855538e+00  4.82910683e+00  6.86576671e+00]]
    x: [[95.21573697]
     [47.17621033]
     [17.36307772]
     [ 7.75517239]]
    P: [[ 1.04992508e+03  2.48771307e-02  9.69862945e+00 -2.81410474e-01]
     [ 2.48771307e-02  1.04992508e+03 -2.81410474e-01  9.69862945e+00]
     [ 9.69862945e+00 -2.81410474e-01  6.84542713e+00  4.84941915e+00]
     [-2.81410474e-01  9.69862945e+00  4.84941915e+00  6.84542713e+00]]
    x: [[97.35723687]
     [47.78006196]
     [17.56463901]
     [ 7.84362432]]
    P: [[ 1.05092509e+03  2.48905087e-02  9.69871060e+00 -2.81719931e-01]
     [ 2.48905087e-02  1.05092509e+03 -2.81719931e-01  9.69871060e+00]
     [ 9.69871060e+00 -2.81719931e-01  6.82588609e+00  4.86893893e+00]
     [-2.81719931e-01  9.69871060e+00  4.86893893e+00  6.82588609e+00]]
    x: [[98.51601377]
     [49.47511511]
     [17.61396398]
     [ 8.18302193]]
    P: [[ 1.05192509e+03  2.49024800e-02  9.69879755e+00 -2.82008596e-01]
     [ 2.49024800e-02  1.05192509e+03 -2.82008596e-01  9.69879755e+00]
     [ 9.69879755e+00 -2.82008596e-01  6.80709712e+00  4.88771132e+00]
     [-2.82008596e-01  9.69879755e+00  4.88771132e+00  6.80709712e+00]]
    x: [[100.41324881]
     [ 51.04731214]
     [ 18.10815128]
     [  8.79382361]]
    P: [[ 1.05292510e+03  2.49132037e-02  9.69888917e+00 -2.82278434e-01]
     [ 2.49132037e-02  1.05292510e+03 -2.82278434e-01  9.69888917e+00]
     [ 9.69888917e+00 -2.82278434e-01  6.78901735e+00  4.90577817e+00]
     [-2.82278434e-01  9.69888917e+00  4.90577817e+00  6.78901735e+00]]
    x: [[102.83862356]
     [ 50.21049133]
     [ 17.6398683 ]
     [  7.89391789]]
    P: [[ 1.05392511e+03  2.49228205e-02  9.69898451e+00 -2.82531204e-01]
     [ 2.49228205e-02  1.05392511e+03 -2.82531204e-01  9.69898451e+00]
     [ 9.69898451e+00 -2.82531204e-01  6.77160714e+00  4.92317830e+00]
     [-2.82531204e-01  9.69898451e+00  4.92317830e+00  6.77160714e+00]]
    x: [[105.73827526]
     [ 50.9685414 ]
     [ 18.4317278 ]
     [  8.47359437]]
    P: [[ 1.05492511e+03  2.49314552e-02  9.69908270e+00 -2.82768476e-01]
     [ 2.49314552e-02  1.05492511e+03 -2.82768476e-01  9.69908270e+00]
     [ 9.69908270e+00 -2.82768476e-01  6.75482981e+00  4.93994777e+00]
     [-2.82768476e-01  9.69908270e+00  4.93994777e+00  6.75482981e+00]]
    x: [[107.42630759]
     [ 51.76801634]
     [ 18.29607437]
     [  8.35709379]]
    P: [[ 1.05592512e+03  2.49392181e-02  9.69918303e+00 -2.82991656e-01]
     [ 2.49392181e-02  1.05592512e+03 -2.82991656e-01  9.69918303e+00]
     [ 9.69918303e+00 -2.82991656e-01  6.73865132e+00  4.95612013e+00]
     [-2.82991656e-01  9.69918303e+00  4.95612013e+00  6.73865132e+00]]
    x: [[109.36615843]
     [ 53.21188977]
     [ 18.69852784]
     [  8.84690176]]
    P: [[ 1.05692512e+03  2.49462070e-02  9.69928486e+00 -2.83202003e-01]
     [ 2.49462070e-02  1.05692512e+03 -2.83202003e-01  9.69928486e+00]
     [ 9.69928486e+00 -2.83202003e-01  6.72304008e+00  4.97172659e+00]
     [-2.83202003e-01  9.69928486e+00  4.97172659e+00  6.72304008e+00]]
    x: [[111.08198105]
     [ 54.72790582]
     [ 18.92722808]
     [  9.21100821]]
    P: [[ 1.05792512e+03  2.49525083e-02  9.69938765e+00 -2.83400643e-01]
     [ 2.49525083e-02  1.05792512e+03 -2.83400643e-01  9.69938765e+00]
     [ 9.69938765e+00 -2.83400643e-01  6.70796667e+00  4.98679627e+00]
     [-2.83400643e-01  9.69938765e+00  4.98679627e+00  6.70796667e+00]]
    x: [[112.90520563]
     [ 55.23520117]
     [ 18.65626969]
     [  8.88169266]]
    P: [[ 1.05892513e+03  2.49581988e-02  9.69949094e+00 -2.83588588e-01]
     [ 2.49581988e-02  1.05892513e+03 -2.83588588e-01  9.69949094e+00]
     [ 9.69949094e+00 -2.83588588e-01  6.69340371e+00  5.00135633e+00]
     [-2.83588588e-01  9.69949094e+00  5.00135633e+00  6.69340371e+00]]
    x: [[114.47677379]
     [ 55.87223182]
     [ 18.31411672]
     [  8.54669306]]
    P: [[ 1.05992513e+03  2.49633462e-02  9.69959432e+00 -2.83766745e-01]
     [ 2.49633462e-02  1.05992513e+03 -2.83766745e-01  9.69959432e+00]
     [ 9.69959432e+00 -2.83766745e-01  6.67932564e+00  5.01543213e+00]
     [-2.83766745e-01  9.69959432e+00  5.01543213e+00  6.67932564e+00]]
    x: [[116.72934962]
     [ 57.27656065]
     [ 18.90648542]
     [  9.16012658]]
    P: [[ 1.06092513e+03  2.49680107e-02  9.69969746e+00 -2.83935927e-01]
     [ 2.49680107e-02  1.06092513e+03 -2.83935927e-01  9.69969746e+00]
     [ 9.69969746e+00 -2.83935927e-01  6.66570861e+00  5.02904740e+00]
     [-2.83935927e-01  9.69969746e+00  5.02904740e+00  6.66570861e+00]]
    x: [[118.62385579]
     [ 58.17717936]
     [ 18.90087363]
     [  9.15140969]]
    P: [[ 1.06192513e+03  2.49722454e-02  9.69980007e+00 -2.84096867e-01]
     [ 2.49722454e-02  1.06192513e+03 -2.84096867e-01  9.69980007e+00]
     [ 9.69980007e+00 -2.84096867e-01  6.65253029e+00  5.04222433e+00]
     [-2.84096867e-01  9.69980007e+00  5.04222433e+00  6.65253029e+00]]
    x: [[119.55511179]
     [ 58.80305352]
     [ 18.07263055]
     [  8.4294467 ]]
    P: [[ 1.06292513e+03  2.49760976e-02  9.69990192e+00 -2.84250223e-01]
     [ 2.49760976e-02  1.06292513e+03 -2.84250223e-01  9.69990192e+00]
     [ 9.69990192e+00 -2.84250223e-01  6.63976980e+00  5.05498375e+00]
     [-2.84250223e-01  9.69990192e+00  5.05498375e+00  6.63976980e+00]]
    x: [[121.14291156]
     [ 58.79095013]
     [ 17.45498155]
     [  7.71248758]]
    P: [[ 1.06392514e+03  2.49796088e-02  9.70000279e+00 -2.84396588e-01]
     [ 2.49796088e-02  1.06392514e+03 -2.84396588e-01  9.70000279e+00]
     [ 9.70000279e+00 -2.84396588e-01  6.62740756e+00  5.06734515e+00]
     [-2.84396588e-01  9.70000279e+00  5.06734515e+00  6.62740756e+00]]
    x: [[123.25436661]
     [ 58.7159906 ]
     [ 17.24997313]
     [  7.3209922 ]]
    P: [[ 1.06492514e+03  2.49828162e-02  9.70010252e+00 -2.84536498e-01]
     [ 2.49828162e-02  1.06492514e+03 -2.84536498e-01  9.70010252e+00]
     [ 9.70010252e+00 -2.84536498e-01  6.61542518e+00  5.07932687e+00]
     [-2.84536498e-01  9.70010252e+00  5.07932687e+00  6.61542518e+00]]
    x: [[125.0090591 ]
     [ 59.58031838]
     [ 17.34276579]
     [  7.42932022]]
    P: [[ 1.06592514e+03  2.49857525e-02  9.70020097e+00 -2.84670434e-01]
     [ 2.49857525e-02  1.06592514e+03 -2.84670434e-01  9.70020097e+00]
     [ 9.70020097e+00 -2.84670434e-01  6.60380543e+00  5.09094612e+00]
     [-2.84670434e-01  9.70020097e+00  5.09094612e+00  6.60380543e+00]]
    x: [[127.26479826]
     [ 60.36047339]
     [ 17.7258676 ]
     [  7.74014747]]
    P: [[ 1.06692514e+03  2.49884468e-02  9.70029804e+00 -2.84798833e-01]
     [ 2.49884468e-02  1.06692514e+03 -2.84798833e-01  9.70029804e+00]
     [ 9.70029804e+00 -2.84798833e-01  6.59253206e+00  5.10221909e+00]
     [-2.84798833e-01  9.70029804e+00  5.10221909e+00  6.59253206e+00]]
    x: [[129.72277938]
     [ 61.73641949]
     [ 18.53148412]
     [  8.53349002]]
    P: [[ 1.06792514e+03  2.49909248e-02  9.70039362e+00 -2.84922091e-01]
     [ 2.49909248e-02  1.06792514e+03 -2.84922091e-01  9.70039362e+00]
     [ 9.70039362e+00 -2.84922091e-01  6.58158981e+00  5.11316103e+00]
     [-2.84922091e-01  9.70039362e+00  5.11316103e+00  6.58158981e+00]]
    x: [[132.00249798]
     [ 62.88528659]
     [ 18.98942682]
     [  8.97243967]]
    P: [[ 1.06892514e+03  2.49932094e-02  9.70048766e+00 -2.85040564e-01]
     [ 2.49932094e-02  1.06892514e+03 -2.85040564e-01  9.70048766e+00]
     [ 9.70048766e+00 -2.85040564e-01  6.57096430e+00  5.12378630e+00]
     [-2.85040564e-01  9.70048766e+00  5.12378630e+00  6.57096430e+00]]
    x: [[133.52999793]
     [ 63.80633866]
     [ 18.7452989 ]
     [  8.78477614]]
    P: [[ 1.06992514e+03  2.49953208e-02  9.70058009e+00 -2.85154577e-01]
     [ 2.49953208e-02  1.06992514e+03 -2.85154577e-01  9.70058009e+00]
     [ 9.70058009e+00 -2.85154577e-01  6.56064197e+00  5.13410844e+00]
     [-2.85154577e-01  9.70058009e+00  5.13410844e+00  6.56064197e+00]]
    x: [[135.91458552]
     [ 64.44297173]
     [ 18.96482174]
     [  8.89839727]]
    P: [[ 1.07092514e+03  2.49972770e-02  9.70067088e+00 -2.85264425e-01]
     [ 2.49972770e-02  1.07092514e+03 -2.85264425e-01  9.70067088e+00]
     [ 9.70067088e+00 -2.85264425e-01  6.55061001e+00  5.14414025e+00]
     [-2.85264425e-01  9.70067088e+00  5.14414025e+00  6.55061001e+00]]
    x: [[138.14505396]
     [ 65.80466054]
     [ 19.45570748]
     [  9.40843061]]
    P: [[ 1.07192514e+03  2.49990939e-02  9.70076000e+00 -2.85370377e-01]
     [ 2.49990939e-02  1.07192514e+03 -2.85370377e-01  9.70076000e+00]
     [ 9.70076000e+00 -2.85370377e-01  6.54085635e+00  5.15389380e+00]
     [-2.85370377e-01  9.70076000e+00  5.15389380e+00  6.54085635e+00]]
    x: [[140.10884422]
     [ 67.58805933]
     [ 19.93382658]
     [  9.99947248]]
    P: [[ 1.07292514e+03  2.50007858e-02  9.70084743e+00 -2.85472676e-01]
     [ 2.50007858e-02  1.07292514e+03 -2.85472676e-01  9.70084743e+00]
     [ 9.70084743e+00 -2.85472676e-01  6.53136956e+00  5.16338050e+00]
     [-2.85472676e-01  9.70084743e+00  5.16338050e+00  6.53136956e+00]]
    x: [[142.36461866]
     [ 68.56231263]
     [ 20.10029213]
     [ 10.12700753]]
    P: [[ 1.07392514e+03  2.50023651e-02  9.70093318e+00 -2.85571545e-01]
     [ 2.50023651e-02  1.07392514e+03 -2.85571545e-01  9.70093318e+00]
     [ 9.70093318e+00 -2.85571545e-01  6.52213883e+00  5.17261116e+00]
     [-2.85571545e-01  9.70093318e+00  5.17261116e+00  6.52213883e+00]]
    x: [[143.25608239]
     [ 70.21407181]
     [ 19.68532078]
     [  9.94638604]]
    P: [[ 1.07492514e+03  2.50038432e-02  9.70101723e+00 -2.85667189e-01]
     [ 2.50038432e-02  1.07492514e+03 -2.85667189e-01  9.70101723e+00]
     [ 9.70101723e+00 -2.85667189e-01  6.51315393e+00  5.18159601e+00]
     [-2.85667189e-01  9.70101723e+00  5.18159601e+00  6.51315393e+00]]
    x: [[145.12494067]
     [ 70.76225626]
     [ 19.36896378]
     [  9.58440005]]
    P: [[ 1.07592514e+03  2.50052298e-02  9.70109960e+00 -2.85759794e-01]
     [ 2.50052298e-02  1.07592514e+03 -2.85759794e-01  9.70109960e+00]
     [ 9.70109960e+00 -2.85759794e-01  6.50440517e+00  5.19034472e+00]
     [-2.85759794e-01  9.70109960e+00  5.19034472e+00  6.50440517e+00]]
    x: [[147.62283332]
     [ 72.23093404]
     [ 20.03754785]
     [ 10.2463921 ]]
    P: [[ 1.07692514e+03  2.50065340e-02  9.70118030e+00 -2.85849532e-01]
     [ 2.50065340e-02  1.07692514e+03 -2.85849532e-01  9.70118030e+00]
     [ 9.70118030e+00 -2.85849532e-01  6.49588336e+00  5.19886650e+00]
     [-2.85849532e-01  9.70118030e+00  5.19886650e+00  6.49588336e+00]]
    x: [[149.3450485 ]
     [ 74.06093798]
     [ 20.29319545]
     [ 10.64138641]]
    P: [[ 1.07792514e+03  2.50077634e-02  9.70125934e+00 -2.85936558e-01]
     [ 2.50077634e-02  1.07792514e+03 -2.85936558e-01  9.70125934e+00]
     [ 9.70125934e+00 -2.85936558e-01  6.48757978e+00  5.20717005e+00]
     [-2.85936558e-01  9.70125934e+00  5.20717005e+00  6.48757978e+00]]
    x: [[151.02272272]
     [ 74.86966074]
     [ 19.91009007]
     [ 10.27046197]]
    P: [[ 1.07892514e+03  2.50089252e-02  9.70133676e+00 -2.86021018e-01]
     [ 2.50089252e-02  1.07892514e+03 -2.86021018e-01  9.70133676e+00]
     [ 9.70133676e+00 -2.86021018e-01  6.47948616e+00  5.21526366e+00]
     [-2.86021018e-01  9.70133676e+00  5.21526366e+00  6.47948616e+00]]
    x: [[152.10675665]
     [ 76.47825151]
     [ 19.61494908]
     [ 10.16138594]]
    P: [[ 1.07992513e+03  2.50100256e-02  9.70141256e+00 -2.86103046e-01]
     [ 2.50100256e-02  1.07992513e+03 -2.86103046e-01  9.70141256e+00]
     [ 9.70141256e+00 -2.86103046e-01  6.47159463e+00  5.22315517e+00]
     [-2.86103046e-01  9.70141256e+00  5.22315517e+00  6.47159463e+00]]
    x: [[154.33302161]
     [ 77.22616952]
     [ 19.64570678]
     [ 10.12634233]]
    P: [[ 1.08092513e+03  2.50110701e-02  9.70148678e+00 -2.86182763e-01]
     [ 2.50110701e-02  1.08092513e+03 -2.86182763e-01  9.70148678e+00]
     [ 9.70148678e+00 -2.86182763e-01  6.46389771e+00  5.23085208e+00]
     [-2.86182763e-01  9.70148678e+00  5.23085208e+00  6.46389771e+00]]
    x: [[156.47798384]
     [ 77.88321094]
     [ 19.56957995]
     [  9.98485155]]
    P: [[ 1.08192513e+03  2.50120637e-02  9.70155945e+00 -2.86260284e-01]
     [ 2.50120637e-02  1.08192513e+03 -2.86260284e-01  9.70155945e+00]
     [ 9.70155945e+00 -2.86260284e-01  6.45638829e+00  5.23836149e+00]
     [-2.86260284e-01  9.70155945e+00  5.23836149e+00  6.45638829e+00]]
    x: [[157.97051087]
     [ 78.81709626]
     [ 19.21693882]
     [  9.68038285]]
    P: [[ 1.08292513e+03  2.50130109e-02  9.70163059e+00 -2.86335714e-01]
     [ 2.50130109e-02  1.08292513e+03 -2.86335714e-01  9.70163059e+00]
     [ 9.70163059e+00 -2.86335714e-01  6.44905960e+00  5.24569017e+00]
     [-2.86335714e-01  9.70163059e+00  5.24569017e+00  6.44905960e+00]]
    x: [[161.35624694]
     [ 79.32537769]
     [ 19.95516296]
     [ 10.18958329]]
    P: [[ 1.08392513e+03  2.50139156e-02  9.70170024e+00 -2.86409150e-01]
     [ 2.50139156e-02  1.08392513e+03 -2.86409150e-01  9.70170024e+00]
     [ 9.70170024e+00 -2.86409150e-01  6.44190520e+00  5.25284456e+00]
     [-2.86409150e-01  9.70170024e+00  5.25284456e+00  6.44190520e+00]]
    x: [[163.94887133]
     [ 79.7384316 ]
     [ 20.02046544]
     [ 10.11335489]]
    P: [[ 1.08492513e+03  2.50147813e-02  9.70176843e+00 -2.86480684e-01]
     [ 2.50147813e-02  1.08492513e+03 -2.86480684e-01  9.70176843e+00]
     [ 9.70176843e+00 -2.86480684e-01  6.43491895e+00  5.25983081e+00]
     [-2.86480684e-01  9.70176843e+00  5.25983081e+00  6.43491895e+00]]
    x: [[164.31729426]
     [ 81.02051881]
     [ 19.06333259]
     [  9.37766102]]
    P: [[ 1.08592513e+03  2.50156113e-02  9.70183520e+00 -2.86550399e-01]
     [ 2.50156113e-02  1.08592513e+03 -2.86550399e-01  9.70183520e+00]
     [ 9.70183520e+00 -2.86550399e-01  6.42809497e+00  5.26665479e+00]
     [-2.86550399e-01  9.70183520e+00  5.26665479e+00  6.42809497e+00]]
    x: [[165.58979941]
     [ 82.32860309]
     [ 18.8419698 ]
     [  9.27171735]]
    P: [[ 1.08692513e+03  2.50164084e-02  9.70190057e+00 -2.86618375e-01]
     [ 2.50164084e-02  1.08692513e+03 -2.86618375e-01  9.70190057e+00]
     [ 9.70190057e+00 -2.86618375e-01  6.42142769e+00  5.27332206e+00]
     [-2.86618375e-01  9.70190057e+00  5.27332206e+00  6.42142769e+00]]
    x: [[167.77582691]
     [ 83.58561476]
     [ 19.23267832]
     [  9.66560875]]
    P: [[ 1.08792513e+03  2.50171750e-02  9.70196458e+00 -2.86684685e-01]
     [ 2.50171750e-02  1.08792513e+03 -2.86684685e-01  9.70196458e+00]
     [ 9.70196458e+00 -2.86684685e-01  6.41491177e+00  5.27983799e+00]
     [-2.86684685e-01  9.70196458e+00  5.27983799e+00  6.41491177e+00]]
    x: [[169.79306767]
     [ 83.55565168]
     [ 18.73341128]
     [  9.04381398]]
    P: [[ 1.08892513e+03  2.50179136e-02  9.70202726e+00 -2.86749398e-01]
     [ 2.50179136e-02  1.08892513e+03 -2.86749398e-01  9.70202726e+00]
     [ 9.70202726e+00 -2.86749398e-01  6.40854210e+00  5.28620765e+00]
     [-2.86749398e-01  9.70202726e+00  5.28620765e+00  6.40854210e+00]]
    x: [[171.85534376]
     [ 85.62520894]
     [ 19.52014875]
     [  9.93902266]]
    P: [[ 1.08992513e+03  2.50186262e-02  9.70208864e+00 -2.86812578e-01]
     [ 2.50186262e-02  1.08992513e+03 -2.86812578e-01  9.70208864e+00]
     [ 9.70208864e+00 -2.86812578e-01  6.40231383e+00  5.29243592e+00]
     [-2.86812578e-01  9.70208864e+00  5.29243592e+00  6.40231383e+00]]
    x: [[174.11391083]
     [ 86.67408645]
     [ 19.75849899]
     [ 10.14972708]]
    P: [[ 1.09092513e+03  2.50193146e-02  9.70214877e+00 -2.86874286e-01]
     [ 2.50193146e-02  1.09092513e+03 -2.86874286e-01  9.70214877e+00]
     [ 9.70214877e+00 -2.86874286e-01  6.39622229e+00  5.29852745e+00]
     [-2.86874286e-01  9.70214877e+00  5.29852745e+00  6.39622229e+00]]
    x: [[176.0635057 ]
     [ 87.96199725]
     [ 19.89544152]
     [ 10.3191906 ]]
    P: [[ 1.09192513e+03  2.50199805e-02  9.70220766e+00 -2.86934578e-01]
     [ 2.50199805e-02  1.09192513e+03 -2.86934578e-01  9.70220766e+00]
     [ 9.70220766e+00 -2.86934578e-01  6.39026304e+00  5.30448671e+00]
     [-2.86934578e-01  9.70220766e+00  5.30448671e+00  6.39026304e+00]]
    x: [[178.16029185]
     [ 89.70956338]
     [ 20.3737886 ]
     [ 10.86295758]]
    P: [[ 1.09292513e+03  2.50206254e-02  9.70226535e+00 -2.86993509e-01]
     [ 2.50206254e-02  1.09292513e+03 -2.86993509e-01  9.70226535e+00]
     [ 9.70226535e+00 -2.86993509e-01  6.38443180e+00  5.31031795e+00]
     [-2.86993509e-01  9.70226535e+00  5.31031795e+00  6.38443180e+00]]
    x: [[180.62106373]
     [ 91.15950273]
     [ 20.8657576 ]
     [ 11.34857026]]
    P: [[ 1.09392513e+03  2.50212507e-02  9.70232188e+00 -2.87051127e-01]
     [ 2.50212507e-02  1.09392513e+03 -2.87051127e-01  9.70232188e+00]
     [ 9.70232188e+00 -2.87051127e-01  6.37872450e+00  5.31602525e+00]
     [-2.87051127e-01  9.70232188e+00  5.31602525e+00  6.37872450e+00]]
    x: [[183.35482235]
     [ 92.03863202]
     [ 21.15639331]
     [ 11.54416275]]
    P: [[ 1.09492513e+03  2.50218576e-02  9.70237727e+00 -2.87107482e-01]
     [ 2.50218576e-02  1.09492513e+03 -2.87107482e-01  9.70237727e+00]
     [ 9.70237727e+00 -2.87107482e-01  6.37313723e+00  5.32161252e+00]
     [-2.87107482e-01  9.70237727e+00  5.32161252e+00  6.37313723e+00]]
    x: [[185.62414061]
     [ 92.99920595]
     [ 21.1495498 ]
     [ 11.5011191 ]]
    P: [[ 1.09592513e+03  2.50224472e-02  9.70243156e+00 -2.87162617e-01]
     [ 2.50224472e-02  1.09592513e+03 -2.87162617e-01  9.70243156e+00]
     [ 9.70243156e+00 -2.87162617e-01  6.36766624e+00  5.32708351e+00]
     [-2.87162617e-01  9.70243156e+00  5.32708351e+00  6.36766624e+00]]
    x: [[187.94495121]
     [ 93.97806067]
     [ 21.19047604]
     [ 11.50316773]]
    P: [[ 1.09692513e+03  2.50230206e-02  9.70248476e+00 -2.87216576e-01]
     [ 2.50230206e-02  1.09692513e+03 -2.87216576e-01  9.70248476e+00]
     [ 9.70248476e+00 -2.87216576e-01  6.36230793e+00  5.33244181e+00]
     [-2.87216576e-01  9.70248476e+00  5.33244181e+00  6.36230793e+00]]
    x: [[189.56754694]
     [ 95.02288966]
     [ 20.7966928 ]
     [ 11.14927879]]
    P: [[ 1.09792513e+03  2.50235787e-02  9.70253692e+00 -2.87269398e-01]
     [ 2.50235787e-02  1.09792513e+03 -2.87269398e-01  9.70253692e+00]
     [ 9.70253692e+00 -2.87269398e-01  6.35705887e+00  5.33769088e+00]
     [-2.87269398e-01  9.70253692e+00  5.33769088e+00  6.35705887e+00]]
    x: [[191.05079417]
     [ 96.87989693]
     [ 20.81955257]
     [ 11.30734072]]
    P: [[ 1.09892513e+03  2.50241223e-02  9.70258806e+00 -2.87321123e-01]
     [ 2.50241223e-02  1.09892513e+03 -2.87321123e-01  9.70258806e+00]
     [ 9.70258806e+00 -2.87321123e-01  6.35191574e+00  5.34283400e+00]
     [-2.87321123e-01  9.70258806e+00  5.34283400e+00  6.35191574e+00]]
    x: [[193.72878449]
     [ 97.35259177]
     [ 20.84374844]
     [ 11.20612916]]
    P: [[ 1.09992512e+03  2.50246522e-02  9.70263820e+00 -2.87371785e-01]
     [ 2.50246522e-02  1.09992512e+03 -2.87371785e-01  9.70263820e+00]
     [ 9.70263820e+00 -2.87371785e-01  6.34687537e+00  5.34787437e+00]
     [-2.87371785e-01  9.70263820e+00  5.34787437e+00  6.34687537e+00]]
    x: [[195.63602216]
     [ 98.4729954 ]
     [ 20.72484816]
     [ 11.1047465 ]]
    P: [[ 1.10092512e+03  2.50251692e-02  9.70268738e+00 -2.87421420e-01]
     [ 2.50251692e-02  1.10092512e+03 -2.87421420e-01  9.70268738e+00]
     [ 9.70268738e+00 -2.87421420e-01  6.34193471e+00  5.35281503e+00]
     [-2.87421420e-01  9.70268738e+00  5.35281503e+00  6.34193471e+00]]
    x: [[197.35515624]
     [ 98.854354  ]
     [ 20.07099925]
     [ 10.41405785]]
    P: [[ 1.10192512e+03  2.50256737e-02  9.70273562e+00 -2.87470061e-01]
     [ 2.50256737e-02  1.10192512e+03 -2.87470061e-01  9.70273562e+00]
     [ 9.70273562e+00 -2.87470061e-01  6.33709084e+00  5.35765891e+00]
     [-2.87470061e-01  9.70273562e+00  5.35765891e+00  6.33709084e+00]]
    x: [[198.91923195]
     [ 99.46563409]
     [ 19.52810109]
     [  9.87241198]]
    P: [[ 1.10292512e+03  2.50261665e-02  9.70278294e+00 -2.87517738e-01]
     [ 2.50261665e-02  1.10292512e+03 -2.87517738e-01  9.70278294e+00]
     [ 9.70278294e+00 -2.87517738e-01  6.33234092e+00  5.36240882e+00]
     [-2.87517738e-01  9.70278294e+00  5.36240882e+00  6.33234092e+00]]
    x: [[200.77738924]
     [ 99.93779686]
     [ 19.16963867]
     [  9.47352402]]
    P: [[ 1.10392512e+03  2.50266481e-02  9.70282937e+00 -2.87564482e-01]
     [ 2.50266481e-02  1.10392512e+03 -2.87564482e-01  9.70282937e+00]
     [ 9.70282937e+00 -2.87564482e-01  6.32768227e+00  5.36706748e+00]
     [-2.87564482e-01  9.70282937e+00  5.36706748e+00  6.32768227e+00]]
    x: [[202.58991867]
     [100.8908019 ]
     [ 19.10304774]
     [  9.41741757]]
    P: [[ 1.10492512e+03  2.50271190e-02  9.70287493e+00 -2.87610321e-01]
     [ 2.50271190e-02  1.10492512e+03 -2.87610321e-01  9.70287493e+00]
     [ 9.70287493e+00 -2.87610321e-01  6.32311227e+00  5.37163748e+00]
     [-2.87610321e-01  9.70287493e+00  5.37163748e+00  6.32311227e+00]]
    x: [[204.59793342]
     [102.44659964]
     [ 19.52073593]
     [  9.88381765]]
    P: [[ 1.10592512e+03  2.50275797e-02  9.70291965e+00 -2.87655282e-01]
     [ 2.50275797e-02  1.10592512e+03 -2.87655282e-01  9.70291965e+00]
     [ 9.70291965e+00 -2.87655282e-01  6.31862841e+00  5.37612134e+00]
     [-2.87655282e-01  9.70291965e+00  5.37612134e+00  6.31862841e+00]]
    x: [[206.89498727]
     [103.51337871]
     [ 19.79613284]
     [ 10.13430026]]
    P: [[ 1.10692512e+03  2.50280305e-02  9.70296354e+00 -2.87699392e-01]
     [ 2.50280305e-02  1.10692512e+03 -2.87699392e-01  9.70296354e+00]
     [ 9.70296354e+00 -2.87699392e-01  6.31422828e+00  5.38052146e+00]
     [-2.87699392e-01  9.70296354e+00  5.38052146e+00  6.31422828e+00]]
    x: [[208.96133201]
     [104.29927664]
     [ 19.72323746]
     [ 10.03230641]]
    P: [[ 1.10792512e+03  2.50284719e-02  9.70300663e+00 -2.87742674e-01]
     [ 2.50284719e-02  1.10792512e+03 -2.87742674e-01  9.70300663e+00]
     [ 9.70300663e+00 -2.87742674e-01  6.30990956e+00  5.38484018e+00]
     [-2.87742674e-01  9.70300663e+00  5.38484018e+00  6.30990956e+00]]
    x: [[211.46548724]
     [104.88490339]
     [ 19.8377309 ]
     [ 10.05969568]]
    P: [[ 1.10892512e+03  2.50289043e-02  9.70304894e+00 -2.87785154e-01]
     [ 2.50289043e-02  1.10892512e+03 -2.87785154e-01  9.70304894e+00]
     [ 9.70304894e+00 -2.87785154e-01  6.30567001e+00  5.38907973e+00]
     [-2.87785154e-01  9.70304894e+00  5.38907973e+00  6.30567001e+00]]
    x: [[214.23330929]
     [105.68637593]
     [ 20.2425971 ]
     [ 10.37469406]]
    P: [[ 1.10992512e+03  2.50293281e-02  9.70309049e+00 -2.87826854e-01]
     [ 2.50293281e-02  1.10992512e+03 -2.87826854e-01  9.70309049e+00]
     [ 9.70309049e+00 -2.87826854e-01  6.30150748e+00  5.39324227e+00]
     [-2.87826854e-01  9.70309049e+00  5.39324227e+00  6.30150748e+00]]
    x: [[216.23325418]
     [106.81334421]
     [ 20.27795309]
     [ 10.42030354]]
    P: [[ 1.11092512e+03  2.50297434e-02  9.70313130e+00 -2.87867796e-01]
     [ 2.50297434e-02  1.11092512e+03 -2.87867796e-01  9.70313130e+00]
     [ 9.70313130e+00 -2.87867796e-01  6.29741988e+00  5.39732987e+00]
     [-2.87867796e-01  9.70313130e+00  5.39732987e+00  6.29741988e+00]]
    x: [[217.93048266]
     [107.39549585]
     [ 19.79281759]
     [  9.92362234]]
    P: [[ 1.11192512e+03  2.50301507e-02  9.70317139e+00 -2.87908001e-01]
     [ 2.50301507e-02  1.11192512e+03 -2.87908001e-01  9.70317139e+00]
     [ 9.70317139e+00 -2.87908001e-01  6.29340520e+00  5.40134454e+00]
     [-2.87908001e-01  9.70317139e+00  5.40134454e+00  6.29340520e+00]]
    x: [[219.16406523]
     [108.43202617]
     [ 19.32217991]
     [  9.52288442]]
    P: [[ 1.11292512e+03  2.50305502e-02  9.70321078e+00 -2.87947489e-01]
     [ 2.50305502e-02  1.11292512e+03 -2.87947489e-01  9.70321078e+00]
     [ 9.70321078e+00 -2.87947489e-01  6.28946152e+00  5.40528822e+00]
     [-2.87947489e-01  9.70321078e+00  5.40528822e+00  6.28946152e+00]]
    x: [[221.4537903 ]
     [110.36497941]
     [ 20.12592372]
     [ 10.38129119]]
    P: [[ 1.11392512e+03  2.50309422e-02  9.70324948e+00 -2.87986280e-01]
     [ 2.50309422e-02  1.11392512e+03 -2.87986280e-01  9.70324948e+00]
     [ 9.70324948e+00 -2.87986280e-01  6.28558697e+00  5.40916278e+00]
     [-2.87986280e-01  9.70324948e+00  5.40916278e+00  6.28558697e+00]]
    x: [[223.37388037]
     [111.37090955]
     [ 20.04585577]
     [ 10.306467  ]]
    P: [[ 1.11492512e+03  2.50313269e-02  9.70328751e+00 -2.88024393e-01]
     [ 2.50313269e-02  1.11492512e+03 -2.88024393e-01  9.70328751e+00]
     [ 9.70328751e+00 -2.88024393e-01  6.28177974e+00  5.41297001e+00]
     [-2.88024393e-01  9.70328751e+00  5.41297001e+00  6.28177974e+00]]
    x: [[225.15394275]
     [112.19416536]
     [ 19.77687588]
     [ 10.03896404]]
    P: [[ 1.11592512e+03  2.50317046e-02  9.70332489e+00 -2.88061844e-01]
     [ 2.50317046e-02  1.11592512e+03 -2.88061844e-01  9.70332489e+00]
     [ 9.70332489e+00 -2.88061844e-01  6.27803809e+00  5.41671165e+00]
     [-2.88061844e-01  9.70332489e+00  5.41671165e+00  6.27803809e+00]]
    x: [[227.46509931]
     [113.03608771]
     [ 19.90455605]
     [ 10.12429865]]
    P: [[ 1.11692512e+03  2.50320755e-02  9.70336164e+00 -2.88098653e-01]
     [ 2.50320755e-02  1.11692512e+03 -2.88098653e-01  9.70336164e+00]
     [ 9.70336164e+00 -2.88098653e-01  6.27436035e+00  5.42038939e+00]
     [-2.88098653e-01  9.70336164e+00  5.42038939e+00  6.27436035e+00]]
    x: [[229.55058749]
     [113.97247619]
     [ 19.92359948]
     [ 10.12884428]]
    P: [[ 1.11792512e+03  2.50324398e-02  9.70339776e+00 -2.88134835e-01]
     [ 2.50324398e-02  1.11792512e+03 -2.88134835e-01  9.70339776e+00]
     [ 9.70339776e+00 -2.88134835e-01  6.27074490e+00  5.42400485e+00]
     [-2.88134835e-01  9.70339776e+00  5.42400485e+00  6.27074490e+00]]
    x: [[231.56211766]
     [114.86650148]
     [ 19.86748695]
     [ 10.06113265]]
    P: [[ 1.11892512e+03  2.50327977e-02  9.70343329e+00 -2.88170407e-01]
     [ 2.50327977e-02  1.11892512e+03 -2.88170407e-01  9.70343329e+00]
     [ 9.70343329e+00 -2.88170407e-01  6.26719015e+00  5.42755959e+00]
     [-2.88170407e-01  9.70343329e+00  5.42755959e+00  6.26719015e+00]]
    x: [[232.90862835]
     [115.29577579]
     [ 19.10897633]
     [  9.30790529]]
    P: [[ 1.11992512e+03  2.50331494e-02  9.70346822e+00 -2.88205384e-01]
     [ 2.50331494e-02  1.11992512e+03 -2.88205384e-01  9.70346822e+00]
     [ 9.70346822e+00 -2.88205384e-01  6.26369461e+00  5.43105514e+00]
     [-2.88205384e-01  9.70346822e+00  5.43105514e+00  6.26369461e+00]]
    x: [[234.77740835]
     [116.42768868]
     [ 19.19767542]
     [  9.41670685]]
    P: [[ 1.12092512e+03  2.50334951e-02  9.70350258e+00 -2.88239781e-01]
     [ 2.50334951e-02  1.12092512e+03 -2.88239781e-01  9.70350258e+00]
     [ 9.70350258e+00 -2.88239781e-01  6.26025679e+00  5.43449296e+00]
     [-2.88239781e-01  9.70350258e+00  5.43449296e+00  6.26025679e+00]]
    x: [[237.26458498]
     [117.16370912]
     [ 19.4540295 ]
     [  9.60969542]]
    P: [[ 1.12192512e+03  2.50338349e-02  9.70353638e+00 -2.88273613e-01]
     [ 2.50338349e-02  1.12192512e+03 -2.88273613e-01  9.70353638e+00]
     [ 9.70353638e+00 -2.88273613e-01  6.25687528e+00  5.43787446e+00]
     [-2.88273613e-01  9.70353638e+00  5.43787446e+00  6.25687528e+00]]
    x: [[238.85730849]
     [119.26468958]
     [ 19.88232542]
     [ 10.15934828]]
    P: [[ 1.12292512e+03  2.50341691e-02  9.70356963e+00 -2.88306893e-01]
     [ 2.50341691e-02  1.12292512e+03 -2.88306893e-01  9.70356963e+00]
     [ 9.70356963e+00 -2.88306893e-01  6.25354871e+00  5.44120103e+00]
     [-2.88306893e-01  9.70356963e+00  5.44120103e+00  6.25354871e+00]]
    x: [[241.32238418]
     [121.02492355]
     [ 20.62993468]
     [ 10.92852657]]
    P: [[ 1.12392512e+03  2.50344977e-02  9.70360235e+00 -2.88339636e-01]
     [ 2.50344977e-02  1.12392512e+03 -2.88339636e-01  9.70360235e+00]
     [ 9.70360235e+00 -2.88339636e-01  6.25027576e+00  5.44447399e+00]
     [-2.88339636e-01  9.70360235e+00  5.44447399e+00  6.25027576e+00]]
    x: [[243.53922964]
     [122.37766616]
     [ 20.88264874]
     [ 11.18972366]]
    P: [[ 1.12492511e+03  2.50348209e-02  9.70363454e+00 -2.88371854e-01]
     [ 2.50348209e-02  1.12492511e+03 -2.88371854e-01  9.70363454e+00]
     [ 9.70363454e+00 -2.88371854e-01  6.24705513e+00  5.44769462e+00]
     [-2.88371854e-01  9.70363454e+00  5.44769462e+00  6.24705513e+00]]
    x: [[245.31147722]
     [123.10010585]
     [ 20.44331267]
     [ 10.74399749]]
    P: [[ 1.12592511e+03  2.50351389e-02  9.70366623e+00 -2.88403560e-01]
     [ 2.50351389e-02  1.12592511e+03 -2.88403560e-01  9.70366623e+00]
     [ 9.70366623e+00 -2.88403560e-01  6.24388558e+00  5.45086417e+00]
     [-2.88403560e-01  9.70366623e+00  5.45086417e+00  6.24388558e+00]]
    x: [[247.67010379]
     [123.25697265]
     [ 20.11716744]
     [ 10.32085791]]
    P: [[ 1.12692511e+03  2.50354517e-02  9.70369741e+00 -2.88434765e-01]
     [ 2.50354517e-02  1.12692511e+03 -2.88434765e-01  9.70369741e+00]
     [ 9.70369741e+00 -2.88434765e-01  6.24076590e+00  5.45398384e+00]
     [-2.88434765e-01  9.70369741e+00  5.45398384e+00  6.24076590e+00]]
    x: [[249.23301265]
     [124.54081451]
     [ 19.96742209]
     [ 10.22584411]]
    P: [[ 1.12792511e+03  2.50357596e-02  9.70372812e+00 -2.88465483e-01]
     [ 2.50357596e-02  1.12792511e+03 -2.88465483e-01  9.70372812e+00]
     [ 9.70372812e+00 -2.88465483e-01  6.23769493e+00  5.45705481e+00]
     [-2.88465483e-01  9.70372812e+00  5.45705481e+00  6.23769493e+00]]
    x: [[251.32312631]
     [126.03190979]
     [ 20.30184037]
     [ 10.58934297]]
    P: [[ 1.12892511e+03  2.50360627e-02  9.70375834e+00 -2.88495724e-01]
     [ 2.50360627e-02  1.12892511e+03 -2.88495724e-01  9.70375834e+00]
     [ 9.70375834e+00 -2.88495724e-01  6.23467154e+00  5.46007820e+00]
     [-2.88495724e-01  9.70375834e+00  5.46007820e+00  6.23467154e+00]]
    x: [[252.66027196]
     [126.78813727]
     [ 19.66841731]
     [  9.98594541]]
    P: [[ 1.12992511e+03  2.50363610e-02  9.70378811e+00 -2.88525499e-01]
     [ 2.50363610e-02  1.12992511e+03 -2.88525499e-01  9.70378811e+00]
     [ 9.70378811e+00 -2.88525499e-01  6.23169463e+00  5.46305512e+00]
     [-2.88525499e-01  9.70378811e+00  5.46305512e+00  6.23169463e+00]]
    x: [[254.12015946]
     [127.17243983]
     [ 18.97616111]
     [  9.28549549]]
    P: [[ 1.13092511e+03  2.50366547e-02  9.70381742e+00 -2.88554820e-01]
     [ 2.50366547e-02  1.13092511e+03 -2.88554820e-01  9.70381742e+00]
     [ 9.70381742e+00 -2.88554820e-01  6.22876313e+00  5.46598662e+00]
     [-2.88554820e-01  9.70381742e+00  5.46598662e+00  6.22876313e+00]]
    x: [[257.00994849]
     [127.98727593]
     [ 19.56362043]
     [  9.78917553]]
    P: [[ 1.13192511e+03  2.50369439e-02  9.70384628e+00 -2.88583696e-01]
     [ 2.50369439e-02  1.13192511e+03 -2.88583696e-01  9.70384628e+00]
     [ 9.70384628e+00 -2.88583696e-01  6.22587601e+00  5.46887374e+00]
     [-2.88583696e-01  9.70384628e+00  5.46887374e+00  6.22587601e+00]]
    x: [[258.75695484]
     [128.68358865]
     [ 19.26084071]
     [  9.48088836]]
    P: [[ 1.13292511e+03  2.50372287e-02  9.70387471e+00 -2.88612137e-01]
     [ 2.50372287e-02  1.13292511e+03 -2.88612137e-01  9.70387471e+00]
     [ 9.70387471e+00 -2.88612137e-01  6.22303227e+00  5.47171747e+00]
     [-2.88612137e-01  9.70387471e+00  5.47171747e+00  6.22303227e+00]]
    x: [[261.16877239]
     [129.52858824]
     [ 19.52045397]
     [  9.69655963]]
    P: [[ 1.13392511e+03  2.50375092e-02  9.70390272e+00 -2.88640154e-01]
     [ 2.50375092e-02  1.13392511e+03 -2.88640154e-01  9.70390272e+00]
     [ 9.70390272e+00 -2.88640154e-01  6.22023095e+00  5.47451879e+00]
     [-2.88640154e-01  9.70390272e+00  5.47451879e+00  6.22023095e+00]]
    x: [[263.11348577]
     [130.447807  ]
     [ 19.48617272]
     [  9.6590854 ]]
    P: [[ 1.13492511e+03  2.50377855e-02  9.70393032e+00 -2.88667756e-01]
     [ 2.50377855e-02  1.13492511e+03 -2.88667756e-01  9.70393032e+00]
     [ 9.70393032e+00 -2.88667756e-01  6.21747110e+00  5.47727865e+00]
     [-2.88667756e-01  9.70393032e+00  5.47727865e+00  6.21747110e+00]]
    x: [[265.93083627]
     [130.58066487]
     [ 19.57089906]
     [  9.61868058]]
    P: [[ 1.13592511e+03  2.50380577e-02  9.70395751e+00 -2.88694952e-01]
     [ 2.50380577e-02  1.13592511e+03 -2.88694952e-01  9.70395751e+00]
     [ 9.70395751e+00 -2.88694952e-01  6.21475180e+00  5.47999794e+00]
     [-2.88694952e-01  9.70395751e+00  5.47999794e+00  6.21475180e+00]]
    x: [[268.69164758]
     [131.09958895]
     [ 19.84045961]
     [  9.79724365]]
    P: [[ 1.13692511e+03  2.50383260e-02  9.70398430e+00 -2.88721751e-01]
     [ 2.50383260e-02  1.13692511e+03 -2.88721751e-01  9.70398430e+00]
     [ 9.70398430e+00 -2.88721751e-01  6.21207217e+00  5.48267757e+00]
     [-2.88721751e-01  9.70398430e+00  5.48267757e+00  6.21207217e+00]]
    x: [[270.24069787]
     [131.4056289 ]
     [ 19.16055892]
     [  9.10004667]]
    P: [[ 1.13792511e+03  2.50385903e-02  9.70401071e+00 -2.88748162e-01]
     [ 2.50385903e-02  1.13792511e+03 -2.88748162e-01  9.70401071e+00]
     [ 9.70401071e+00 -2.88748162e-01  6.20943135e+00  5.48531839e+00]
     [-2.88748162e-01  9.70401071e+00  5.48531839e+00  6.20943135e+00]]
    x: [[272.46636946]
     [133.08871437]
     [ 19.81629917]
     [  9.78912974]]
    P: [[ 1.13892511e+03  2.50388508e-02  9.70403674e+00 -2.88774192e-01]
     [ 2.50388508e-02  1.13892511e+03 -2.88774192e-01  9.70403674e+00]
     [ 9.70403674e+00 -2.88774192e-01  6.20682850e+00  5.48792124e+00]
     [-2.88774192e-01  9.70403674e+00  5.48792124e+00  6.20682850e+00]]
    x: [[274.84618399]
     [133.37809086]
     [ 19.67420147]
     [  9.56933767]]
    P: [[ 1.13992511e+03  2.50391075e-02  9.70406239e+00 -2.88799851e-01]
     [ 2.50391075e-02  1.13992511e+03 -2.88799851e-01  9.70406239e+00]
     [ 9.70406239e+00 -2.88799851e-01  6.20426281e+00  5.49048694e+00]
     [-2.88799851e-01  9.70406239e+00  5.49048694e+00  6.20426281e+00]]
    x: [[277.24172666]
     [134.38980086]
     [ 19.9873463 ]
     [  9.85600405]]
    P: [[ 1.14092511e+03  2.50393606e-02  9.70408768e+00 -2.88825146e-01]
     [ 2.50393606e-02  1.14092511e+03 -2.88825146e-01  9.70408768e+00]
     [ 9.70408768e+00 -2.88825146e-01  6.20173348e+00  5.49301626e+00]
     [-2.88825146e-01  9.70408768e+00  5.49301626e+00  6.20173348e+00]]
    x: [[279.55439941]
     [135.07775579]
     [ 20.01899953]
     [  9.84458801]]
    P: [[ 1.14192511e+03  2.50396101e-02  9.70411262e+00 -2.88850084e-01]
     [ 2.50396101e-02  1.14192511e+03 -2.88850084e-01  9.70411262e+00]
     [ 9.70411262e+00 -2.88850084e-01  6.19923976e+00  5.49550999e+00]
     [-2.88850084e-01  9.70411262e+00  5.49550999e+00  6.19923976e+00]]
    x: [[281.29236667]
     [135.67474706]
     [ 19.61876502]
     [  9.4357147 ]]
    P: [[ 1.14292511e+03  2.50398561e-02  9.70413721e+00 -2.88874674e-01]
     [ 2.50398561e-02  1.14292511e+03 -2.88874674e-01  9.70413721e+00]
     [ 9.70413721e+00 -2.88874674e-01  6.19678088e+00  5.49796886e+00]
     [-2.88874674e-01  9.70413721e+00  5.49796886e+00  6.19678088e+00]]
    x: [[283.38085239]
     [136.86636209]
     [ 19.84722769]
     [  9.67261031]]
    P: [[ 1.14392511e+03  2.50400987e-02  9.70416145e+00 -2.88898923e-01]
     [ 2.50400987e-02  1.14392511e+03 -2.88898923e-01  9.70416145e+00]
     [ 9.70416145e+00 -2.88898923e-01  6.19435614e+00  5.50039361e+00]
     [-2.88898923e-01  9.70416145e+00  5.50039361e+00  6.19435614e+00]]
    x: [[285.20889904]
     [137.0709581 ]
     [ 19.2971645 ]
     [  9.08075478]]
    P: [[ 1.14492511e+03  2.50403379e-02  9.70418536e+00 -2.88922837e-01]
     [ 2.50403379e-02  1.14492511e+03 -2.88922837e-01  9.70418536e+00]
     [ 9.70418536e+00 -2.88922837e-01  6.19196481e+00  5.50278493e+00]
     [-2.88922837e-01  9.70418536e+00  5.50278493e+00  6.19196481e+00]]
    x: [[287.47026617]
     [137.9533626 ]
     [ 19.49943109]
     [  9.25854728]]
    P: [[ 1.14592511e+03  2.50405739e-02  9.70420895e+00 -2.88946424e-01]
     [ 2.50405739e-02  1.14592511e+03 -2.88946424e-01  9.70420895e+00]
     [ 9.70420895e+00 -2.88946424e-01  6.18960622e+00  5.50514352e+00]
     [-2.88946424e-01  9.70420895e+00  5.50514352e+00  6.18960622e+00]]
    x: [[288.78873655]
     [138.97777238]
     [ 19.14363021]
     [  8.95240816]]
    P: [[ 1.14692511e+03  2.50408066e-02  9.70423221e+00 -2.88969690e-01]
     [ 2.50408066e-02  1.14692511e+03 -2.88969690e-01  9.70423221e+00]
     [ 9.70423221e+00 -2.88969690e-01  6.18727970e+00  5.50747005e+00]
     [-2.88969690e-01  9.70423221e+00  5.50747005e+00  6.18727970e+00]]
    x: [[291.23828343]
     [139.54738766]
     [ 19.30285838]
     [  9.05347354]]
    P: [[ 1.14792511e+03  2.50410362e-02  9.70425516e+00 -2.88992641e-01]
     [ 2.50410362e-02  1.14792511e+03 -2.88992641e-01  9.70425516e+00]
     [ 9.70425516e+00 -2.88992641e-01  6.18498460e+00  5.50976515e+00]
     [-2.88992641e-01  9.70425516e+00  5.50976515e+00  6.18498460e+00]]
    x: [[293.39793907]
     [139.92799147]
     [ 19.14472152]
     [  8.84472504]]
    P: [[ 1.14892511e+03  2.50412627e-02  9.70427781e+00 -2.89015285e-01]
     [ 2.50412627e-02  1.14892511e+03 -2.89015285e-01  9.70427781e+00]
     [ 9.70427781e+00 -2.89015285e-01  6.18272028e+00  5.51202947e+00]
     [-2.89015285e-01  9.70427781e+00  5.51202947e+00  6.18272028e+00]]
    x: [[295.5277805 ]
     [140.09182759]
     [ 18.86212417]
     [  8.49972731]]
    P: [[ 1.14992511e+03  2.50414861e-02  9.70430015e+00 -2.89037627e-01]
     [ 2.50414861e-02  1.14992511e+03 -2.89037627e-01  9.70430015e+00]
     [ 9.70430015e+00 -2.89037627e-01  6.18048613e+00  5.51426361e+00]
     [-2.89037627e-01  9.70430015e+00  5.51426361e+00  6.18048613e+00]]
    x: [[297.83934283]
     [141.06590359]
     [ 19.21333865]
     [  8.83099168]]
    P: [[ 1.15092511e+03  2.50417066e-02  9.70432219e+00 -2.89059673e-01]
     [ 2.50417066e-02  1.15092511e+03 -2.89059673e-01  9.70432219e+00]
     [ 9.70432219e+00 -2.89059673e-01  6.17828156e+00  5.51646819e+00]
     [-2.89059673e-01  9.70432219e+00  5.51646819e+00  6.17828156e+00]]
    x: [[300.24887258]
     [141.20891654]
     [ 19.09730548]
     [  8.63415047]]
    P: [[ 1.15192511e+03  2.50419242e-02  9.70434395e+00 -2.89081430e-01]
     [ 2.50419242e-02  1.15192511e+03 -2.89081430e-01  9.70434395e+00]
     [ 9.70434395e+00 -2.89081430e-01  6.17610597e+00  5.51864377e+00]
     [-2.89081430e-01  9.70434395e+00  5.51864377e+00  6.17610597e+00]]
    x: [[301.21969698]
     [142.4528    ]
     [ 18.70736935]
     [  8.33044798]]
    P: [[ 1.15292511e+03  2.50421390e-02  9.70436542e+00 -2.89102902e-01]
     [ 2.50421390e-02  1.15292511e+03 -2.89102902e-01  9.70436542e+00]
     [ 9.70436542e+00 -2.89102902e-01  6.17395880e+00  5.52079094e+00]
     [-2.89102902e-01  9.70436542e+00  5.52079094e+00  6.17395880e+00]]
    x: [[302.23835267]
     [144.28519957]
     [ 18.73872243]
     [  8.48202418]]
    P: [[ 1.15392511e+03  2.50423509e-02  9.70438661e+00 -2.89124095e-01]
     [ 2.50423509e-02  1.15392511e+03 -2.89124095e-01  9.70438661e+00]
     [ 9.70438661e+00 -2.89124095e-01  6.17183951e+00  5.52291024e+00]
     [-2.89124095e-01  9.70438661e+00  5.52291024e+00  6.17183951e+00]]
    x: [[303.54403416]
     [145.55214972]
     [ 18.61407371]
     [  8.4210489 ]]
    P: [[ 1.15492511e+03  2.50425602e-02  9.70440753e+00 -2.89145015e-01]
     [ 2.50425602e-02  1.15492511e+03 -2.89145015e-01  9.70440753e+00]
     [ 9.70440753e+00 -2.89145015e-01  6.16974754e+00  5.52500221e+00]
     [-2.89145015e-01  9.70440753e+00  5.52500221e+00  6.16974754e+00]]
    x: [[305.44002931]
     [146.45041006]
     [ 18.66973877]
     [  8.47809651]]
    P: [[ 1.15592511e+03  2.50427667e-02  9.70442818e+00 -2.89165667e-01]
     [ 2.50427667e-02  1.15592511e+03 -2.89165667e-01  9.70442818e+00]
     [ 9.70442818e+00 -2.89165667e-01  6.16768237e+00  5.52706738e+00]
     [-2.89165667e-01  9.70442818e+00  5.52706738e+00  6.16768237e+00]]
    x: [[306.39575154]
     [147.4410872 ]
     [ 18.15896099]
     [  8.03446007]]
    P: [[ 1.15692511e+03  2.50429706e-02  9.70444857e+00 -2.89186056e-01]
     [ 2.50429706e-02  1.15692511e+03 -2.89186056e-01  9.70444857e+00]
     [ 9.70444857e+00 -2.89186056e-01  6.16564349e+00  5.52910625e+00]
     [-2.89186056e-01  9.70444857e+00  5.52910625e+00  6.16564349e+00]]
    x: [[308.22631966]
     [147.89565861]
     [ 17.96290819]
     [  7.815398  ]]
    P: [[ 1.15792511e+03  2.50431719e-02  9.70446870e+00 -2.89206187e-01]
     [ 2.50431719e-02  1.15792511e+03 -2.89206187e-01  9.70446870e+00]
     [ 9.70446870e+00 -2.89206187e-01  6.16363041e+00  5.53111934e+00]
     [-2.89206187e-01  9.70446870e+00  5.53111934e+00  6.16363041e+00]]
    x: [[310.60864575]
     [149.56495687]
     [ 18.86872372]
     [  8.74018982]]
    P: [[ 1.15892511e+03  2.50433707e-02  9.70448858e+00 -2.89226065e-01]
     [ 2.50433707e-02  1.15892511e+03 -2.89226065e-01  9.70448858e+00]
     [ 9.70448858e+00 -2.89226065e-01  6.16164263e+00  5.53310712e+00]
     [-2.89226065e-01  9.70448858e+00  5.53310712e+00  6.16164263e+00]]
    x: [[312.797664  ]
     [151.11325285]
     [ 19.46350475]
     [  9.35822906]]
    P: [[ 1.15992511e+03  2.50435671e-02  9.70450821e+00 -2.89245694e-01]
     [ 2.50435671e-02  1.15992511e+03 -2.89245694e-01  9.70450821e+00]
     [ 9.70450821e+00 -2.89245694e-01  6.15967968e+00  5.53507006e+00]
     [-2.89245694e-01  9.70450821e+00  5.53507006e+00  6.15967968e+00]]
    x: [[314.47925987]
     [152.31890977]
     [ 19.4500572 ]
     [  9.37798576]]
    P: [[ 1.16092511e+03  2.50437609e-02  9.70452760e+00 -2.89265080e-01]
     [ 2.50437609e-02  1.16092511e+03 -2.89265080e-01  9.70452760e+00]
     [ 9.70452760e+00 -2.89265080e-01  6.15774110e+00  5.53700864e+00]
     [-2.89265080e-01  9.70452760e+00  5.53700864e+00  6.15774110e+00]]
    x: [[316.30737657]
     [152.55765257]
     [ 18.96128054]
     [  8.85327288]]
    P: [[ 1.16192511e+03  2.50439524e-02  9.70454674e+00 -2.89284227e-01]
     [ 2.50439524e-02  1.16192511e+03 -2.89284227e-01  9.70454674e+00]
     [ 9.70454674e+00 -2.89284227e-01  6.15582645e+00  5.53892330e+00]
     [-2.89284227e-01  9.70454674e+00  5.53892330e+00  6.15582645e+00]]
    x: [[318.74442689]
     [152.93447047]
     [ 19.01360179]
     [  8.84121183]]
    P: [[ 1.16292511e+03  2.50441415e-02  9.70456565e+00 -2.89303139e-01]
     [ 2.50441415e-02  1.16292511e+03 -2.89303139e-01  9.70456565e+00]
     [ 9.70456565e+00 -2.89303139e-01  6.15393526e+00  5.54081448e+00]
     [-2.89303139e-01  9.70456565e+00  5.54081448e+00  6.15393526e+00]]
    x: [[320.67260923]
     [153.90667727]
     [ 19.0830978 ]
     [  8.91444341]]
    P: [[ 1.16392511e+03  2.50443284e-02  9.70458433e+00 -2.89321820e-01]
     [ 2.50443284e-02  1.16392511e+03 -2.89321820e-01  9.70458433e+00]
     [ 9.70458433e+00 -2.89321820e-01  6.15206713e+00  5.54268261e+00]
     [-2.89321820e-01  9.70458433e+00  5.54268261e+00  6.15206713e+00]]
    x: [[321.87890059]
     [154.69958803]
     [ 18.56762535]
     [  8.4355458 ]]
    P: [[ 1.16492511e+03  2.50445129e-02  9.70460279e+00 -2.89340275e-01]
     [ 2.50445129e-02  1.16492511e+03 -2.89340275e-01  9.70460279e+00]
     [ 9.70460279e+00 -2.89340275e-01  6.15022163e+00  5.54452811e+00]
     [-2.89340275e-01  9.70460279e+00  5.54452811e+00  6.15022163e+00]]
    x: [[324.09678558]
     [155.97194736]
     [ 19.0561732 ]
     [  8.92817089]]
    P: [[ 1.16592510e+03  2.50446952e-02  9.70462102e+00 -2.89358508e-01]
     [ 2.50446952e-02  1.16592510e+03 -2.89358508e-01  9.70462102e+00]
     [ 9.70462102e+00 -2.89358508e-01  6.14839835e+00  5.54635139e+00]
     [-2.89358508e-01  9.70462102e+00  5.54635139e+00  6.14839835e+00]]
    x: [[325.75235269]
     [156.45739065]
     [ 18.65258848]
     [  8.51516561]]
    P: [[ 1.16692510e+03  2.50448754e-02  9.70463904e+00 -2.89376523e-01]
     [ 2.50448754e-02  1.16692510e+03 -2.89376523e-01  9.70463904e+00]
     [ 9.70463904e+00 -2.89376523e-01  6.14659690e+00  5.54815285e+00]
     [-2.89376523e-01  9.70463904e+00  5.54815285e+00  6.14659690e+00]]
    x: [[327.78379535]
     [157.77357608]
     [ 19.0354978 ]
     [  8.9158419 ]]
    P: [[ 1.16792510e+03  2.50450534e-02  9.70465684e+00 -2.89394323e-01]
     [ 2.50450534e-02  1.16792510e+03 -2.89394323e-01  9.70465684e+00]
     [ 9.70465684e+00 -2.89394323e-01  6.14481687e+00  5.54993287e+00]
     [-2.89394323e-01  9.70465684e+00  5.54993287e+00  6.14481687e+00]]
    x: [[329.79521009]
     [158.50216451]
     [ 19.00927231]
     [  8.87358913]]
    P: [[ 1.16892510e+03  2.50452293e-02  9.70467443e+00 -2.89411913e-01]
     [ 2.50452293e-02  1.16892510e+03 -2.89411913e-01  9.70467443e+00]
     [ 9.70467443e+00 -2.89411913e-01  6.14305790e+00  5.55169184e+00]
     [-2.89411913e-01  9.70467443e+00  5.55169184e+00  6.14305790e+00]]
    x: [[330.61920795]
     [159.93295864]
     [ 18.6302865 ]
     [  8.58991889]]
    P: [[ 1.16992510e+03  2.50454031e-02  9.70469181e+00 -2.89429296e-01]
     [ 2.50454031e-02  1.16992510e+03 -2.89429296e-01  9.70469181e+00]
     [ 9.70469181e+00 -2.89429296e-01  6.14131961e+00  5.55343013e+00]
     [-2.89429296e-01  9.70469181e+00  5.55343013e+00  6.14131961e+00]]
    x: [[331.87560282]
     [161.58105291]
     [ 18.70279813]
     [  8.74405252]]
    P: [[ 1.17092510e+03  2.50455749e-02  9.70470899e+00 -2.89446476e-01]
     [ 2.50455749e-02  1.17092510e+03 -2.89446476e-01  9.70470899e+00]
     [ 9.70470899e+00 -2.89446476e-01  6.13960164e+00  5.55514810e+00]
     [-2.89446476e-01  9.70470899e+00  5.55514810e+00  6.13960164e+00]]
    x: [[333.65858202]
     [163.01425449]
     [ 18.97683996]
     [  9.05565813]]
    P: [[ 1.17192510e+03  2.50457447e-02  9.70472597e+00 -2.89463456e-01]
     [ 2.50457447e-02  1.17192510e+03 -2.89463456e-01  9.70472597e+00]
     [ 9.70472597e+00 -2.89463456e-01  6.13790363e+00  5.55684611e+00]
     [-2.89463456e-01  9.70472597e+00  5.55684611e+00  6.13790363e+00]]
    x: [[335.83225705]
     [163.5311082 ]
     [ 18.9260453 ]
     [  8.96644132]]
    P: [[ 1.17292510e+03  2.50459126e-02  9.70474275e+00 -2.89480240e-01]
     [ 2.50459126e-02  1.17292510e+03 -2.89480240e-01  9.70474275e+00]
     [ 9.70474275e+00 -2.89480240e-01  6.13622525e+00  5.55852450e+00]
     [-2.89480240e-01  9.70474275e+00  5.55852450e+00  6.13622525e+00]]
    x: [[338.50217466]
     [164.96152476]
     [ 19.74729689]
     [  9.77369632]]
    P: [[ 1.17392510e+03  2.50460785e-02  9.70475934e+00 -2.89496831e-01]
     [ 2.50460785e-02  1.17392510e+03 -2.89496831e-01  9.70475934e+00]
     [ 9.70475934e+00 -2.89496831e-01  6.13456614e+00  5.56018360e+00]
     [-2.89496831e-01  9.70475934e+00  5.56018360e+00  6.13456614e+00]]
    x: [[340.12659957]
     [166.04624744]
     [ 19.5833351 ]
     [  9.63588641]]
    P: [[ 1.17492510e+03  2.50462425e-02  9.70477575e+00 -2.89513232e-01]
     [ 2.50462425e-02  1.17492510e+03 -2.89513232e-01  9.70477575e+00]
     [ 9.70477575e+00 -2.89513232e-01  6.13292599e+00  5.56182376e+00]
     [-2.89513232e-01  9.70477575e+00  5.56182376e+00  6.13292599e+00]]
    x: [[343.5641442 ]
     [167.12752508]
     [ 20.61377395]
     [ 10.58896605]]
    P: [[ 1.17592510e+03  2.50464047e-02  9.70479196e+00 -2.89529448e-01]
     [ 2.50464047e-02  1.17592510e+03 -2.89529448e-01  9.70479196e+00]
     [ 9.70479196e+00 -2.89529448e-01  6.13130446e+00  5.56344529e+00]
     [-2.89529448e-01  9.70479196e+00  5.56344529e+00  6.13130446e+00]]
    x: [[345.20761031]
     [168.09804057]
     [ 20.29003089]
     [ 10.28384051]]
    P: [[ 1.17692510e+03  2.50465650e-02  9.70480799e+00 -2.89545480e-01]
     [ 2.50465650e-02  1.17692510e+03 -2.89545480e-01  9.70480799e+00]
     [ 9.70480799e+00 -2.89545480e-01  6.12970124e+00  5.56504850e+00]
     [-2.89545480e-01  9.70480799e+00  5.56504850e+00  6.12970124e+00]]
    x: [[346.42373022]
     [168.53904372]
     [ 19.41405867]
     [  9.42053696]]
    P: [[ 1.17792510e+03  2.50467235e-02  9.70482384e+00 -2.89561332e-01]
     [ 2.50467235e-02  1.17792510e+03 -2.89561332e-01  9.70482384e+00]
     [ 9.70482384e+00 -2.89561332e-01  6.12811603e+00  5.56663372e+00]
     [-2.89561332e-01  9.70482384e+00  5.56663372e+00  6.12811603e+00]]
    x: [[348.34806017]
     [169.49035926]
     [ 19.40847003]
     [  9.4164197 ]]
    P: [[ 1.17892510e+03  2.50468803e-02  9.70483952e+00 -2.89577007e-01]
     [ 2.50468803e-02  1.17892510e+03 -2.89577007e-01  9.70483952e+00]
     [ 9.70483952e+00 -2.89577007e-01  6.12654852e+00  5.56820123e+00]
     [-2.89577007e-01  9.70483952e+00  5.56820123e+00  6.12654852e+00]]
    x: [[350.43135204]
     [170.65039737]
     [ 19.63046158]
     [  9.64263076]]
    P: [[ 1.17992510e+03  2.50470353e-02  9.70485502e+00 -2.89592508e-01]
     [ 2.50470353e-02  1.17992510e+03 -2.89592508e-01  9.70485502e+00]
     [ 9.70485502e+00 -2.89592508e-01  6.12499841e+00  5.56975133e+00]
     [-2.89592508e-01  9.70485502e+00  5.56975133e+00  6.12499841e+00]]
    x: [[352.09805785]
     [171.56624772]
     [ 19.40950289]
     [  9.43536974]]
    P: [[ 1.18092510e+03  2.50471886e-02  9.70487035e+00 -2.89607838e-01]
     [ 2.50471886e-02  1.18092510e+03 -2.89607838e-01  9.70487035e+00]
     [ 9.70487035e+00 -2.89607838e-01  6.12346543e+00  5.57128431e+00]
     [-2.89607838e-01  9.70487035e+00  5.57128431e+00  6.12346543e+00]]
    x: [[353.89668686]
     [173.22122586]
     [ 19.73950137]
     [  9.81227824]]
    P: [[ 1.18192510e+03  2.50473402e-02  9.70488551e+00 -2.89622999e-01]
     [ 2.50473402e-02  1.18192510e+03 -2.89622999e-01  9.70488551e+00]
     [ 9.70488551e+00 -2.89622999e-01  6.12194928e+00  5.57280046e+00]
     [-2.89622999e-01  9.70488551e+00  5.57280046e+00  6.12194928e+00]]
    x: [[355.94460055]
     [173.70965421]
     [ 19.49486748]
     [  9.53667369]]
    P: [[ 1.18292510e+03  2.50474902e-02  9.70490051e+00 -2.89637995e-01]
     [ 2.50474902e-02  1.18292510e+03 -2.89637995e-01  9.70490051e+00]
     [ 9.70490051e+00 -2.89637995e-01  6.12044969e+00  5.57430005e+00]
     [-2.89637995e-01  9.70490051e+00  5.57430005e+00  6.12044969e+00]]
    x: [[357.84810306]
     [173.81005723]
     [ 18.95832362]
     [  8.95625591]]
    P: [[ 1.18392510e+03  2.50476385e-02  9.70491534e+00 -2.89652828e-01]
     [ 2.50476385e-02  1.18392510e+03 -2.89652828e-01  9.70491534e+00]
     [ 9.70491534e+00 -2.89652828e-01  6.11896640e+00  5.57578335e+00]
     [-2.89652828e-01  9.70491534e+00  5.57578335e+00  6.11896640e+00]]
    x: [[359.70896889]
     [174.85903655]
     [ 19.02675858]
     [  9.03487034]]
    P: [[ 1.18492510e+03  2.50477852e-02  9.70493001e+00 -2.89667501e-01]
     [ 2.50477852e-02  1.18492510e+03 -2.89667501e-01  9.70493001e+00]
     [ 9.70493001e+00 -2.89667501e-01  6.11749913e+00  5.57725062e+00]
     [-2.89667501e-01  9.70493001e+00  5.57725062e+00  6.11749913e+00]]
    x: [[361.78775733]
     [176.28746597]
     [ 19.45277357]
     [  9.47963962]]
    P: [[ 1.18592510e+03  2.50479304e-02  9.70494453e+00 -2.89682016e-01]
     [ 2.50479304e-02  1.18592510e+03 -2.89682016e-01  9.70494453e+00]
     [ 9.70494453e+00 -2.89682016e-01  6.11604763e+00  5.57870212e+00]
     [-2.89682016e-01  9.70494453e+00  5.57870212e+00  6.11604763e+00]]
    x: [[364.02647851]
     [177.32516427]
     [ 19.6961942 ]
     [  9.7121667 ]]
    P: [[ 1.18692510e+03  2.50480740e-02  9.70495889e+00 -2.89696376e-01]
     [ 2.50480740e-02  1.18692510e+03 -2.89696376e-01  9.70495889e+00]
     [ 9.70495889e+00 -2.89696376e-01  6.11461164e+00  5.58013810e+00]
     [-2.89696376e-01  9.70495889e+00  5.58013810e+00  6.11461164e+00]]
    x: [[365.94969147]
     [178.29153059]
     [ 19.6632558 ]
     [  9.68143873]]
    P: [[ 1.18792510e+03  2.50482160e-02  9.70497310e+00 -2.89710583e-01]
     [ 2.50482160e-02  1.18792510e+03 -2.89710583e-01  9.70497310e+00]
     [ 9.70497310e+00 -2.89710583e-01  6.11319093e+00  5.58155882e+00]
     [-2.89710583e-01  9.70497310e+00  5.58155882e+00  6.11319093e+00]]
    x: [[367.67525808]
     [179.51293254]
     [ 19.65794917]
     [  9.70227057]]
    P: [[ 1.18892510e+03  2.50483566e-02  9.70498715e+00 -2.89724640e-01]
     [ 2.50483566e-02  1.18892510e+03 -2.89724640e-01  9.70498715e+00]
     [ 9.70498715e+00 -2.89724640e-01  6.11178524e+00  5.58296451e+00]
     [-2.89724640e-01  9.70498715e+00  5.58296451e+00  6.11178524e+00]]
    x: [[368.59937456]
     [180.64195107]
     [ 19.07803846]
     [  9.18554248]]
    P: [[ 1.18992510e+03  2.50484957e-02  9.70500106e+00 -2.89738549e-01]
     [ 2.50484957e-02  1.18992510e+03 -2.89738549e-01  9.70500106e+00]
     [ 9.70500106e+00 -2.89738549e-01  6.11039434e+00  5.58435541e+00]
     [-2.89738549e-01  9.70500106e+00  5.58435541e+00  6.11039434e+00]]
    x: [[371.02231402]
     [182.18466799]
     [ 19.78274871]
     [  9.89596096]]
    P: [[ 1.19092510e+03  2.50486333e-02  9.70501483e+00 -2.89752312e-01]
     [ 2.50486333e-02  1.19092510e+03 -2.89752312e-01  9.70501483e+00]
     [ 9.70501483e+00 -2.89752312e-01  6.10901799e+00  5.58573175e+00]
     [-2.89752312e-01  9.70501483e+00  5.58573175e+00  6.10901799e+00]]
    x: [[373.08351357]
     [182.601293  ]
     [ 19.49548465]
     [  9.57453566]]
    P: [[ 1.19192510e+03  2.50487695e-02  9.70502845e+00 -2.89765933e-01]
     [ 2.50487695e-02  1.19192510e+03 -2.89765933e-01  9.70502845e+00]
     [ 9.70502845e+00 -2.89765933e-01  6.10765598e+00  5.58709377e+00]
     [-2.89765933e-01  9.70502845e+00  5.58709377e+00  6.10765598e+00]]
    x: [[374.92806737]
     [183.5143166 ]
     [ 19.40111503]
     [  9.48330411]]
    P: [[ 1.19292510e+03  2.50489043e-02  9.70504192e+00 -2.89779412e-01]
     [ 2.50489043e-02  1.19292510e+03 -2.89779412e-01  9.70504192e+00]
     [ 9.70504192e+00 -2.89779412e-01  6.10630807e+00  5.58844168e+00]
     [-2.89779412e-01  9.70504192e+00  5.58844168e+00  6.10630807e+00]]
    x: [[376.80702943]
     [184.22891132]
     [ 19.2224239 ]
     [  9.29571678]]
    P: [[ 1.19392510e+03  2.50490377e-02  9.70505526e+00 -2.89792752e-01]
     [ 2.50490377e-02  1.19392510e+03 -2.89792752e-01  9.70505526e+00]
     [ 9.70505526e+00 -2.89792752e-01  6.10497405e+00  5.58977570e+00]
     [-2.89792752e-01  9.70505526e+00  5.58977570e+00  6.10497405e+00]]
    x: [[379.05668658]
     [186.09801212]
     [ 19.99356767]
     [ 10.09825103]]
    P: [[ 1.19492510e+03  2.50491698e-02  9.70506847e+00 -2.89805955e-01]
     [ 2.50491698e-02  1.19492510e+03 -2.89805955e-01  9.70506847e+00]
     [ 9.70506847e+00 -2.89805955e-01  6.10365370e+00  5.59109604e+00]
     [-2.89805955e-01  9.70506847e+00  5.59109604e+00  6.10365370e+00]]
    x: [[380.69704252]
     [187.74228064]
     [ 20.13928909]
     [ 10.29465838]]
    P: [[ 1.19592510e+03  2.50493005e-02  9.70508154e+00 -2.89819024e-01]
     [ 2.50493005e-02  1.19592510e+03 -2.89819024e-01  9.70508154e+00]
     [ 9.70508154e+00 -2.89819024e-01  6.10234683e+00  5.59240292e+00]
     [-2.89819024e-01  9.70508154e+00  5.59240292e+00  6.10234683e+00]]
    x: [[381.76119067]
     [188.5372575 ]
     [ 19.38564047]
     [  9.57731899]]
    P: [[ 1.19692510e+03  2.50494298e-02  9.70509447e+00 -2.89831960e-01]
     [ 2.50494298e-02  1.19692510e+03 -2.89831960e-01  9.70509447e+00]
     [ 9.70509447e+00 -2.89831960e-01  6.10105321e+00  5.59369654e+00]
     [-2.89831960e-01  9.70509447e+00  5.59369654e+00  6.10105321e+00]]
    x: [[383.22304243]
     [189.71040573]
     [ 19.2058839 ]
     [  9.43251841]]
    P: [[ 1.19792510e+03  2.50495579e-02  9.70510728e+00 -2.89844766e-01]
     [ 2.50495579e-02  1.19792510e+03 -2.89844766e-01  9.70510728e+00]
     [ 9.70510728e+00 -2.89844766e-01  6.09977265e+00  5.59497709e+00]
     [-2.89844766e-01  9.70510728e+00  5.59497709e+00  6.09977265e+00]]
    x: [[385.13915458]
     [190.6783313 ]
     [ 19.21769519]
     [  9.44579452]]
    P: [[ 1.19892510e+03  2.50496846e-02  9.70511996e+00 -2.89857443e-01]
     [ 2.50496846e-02  1.19892510e+03 -2.89857443e-01  9.70511996e+00]
     [ 9.70511996e+00 -2.89857443e-01  6.09850496e+00  5.59624479e+00]
     [-2.89857443e-01  9.70511996e+00  5.59624479e+00  6.09850496e+00]]
    x: [[387.70923368]
     [191.59946559]
     [ 19.62256284]
     [  9.81707444]]
    P: [[ 1.19992510e+03  2.50498101e-02  9.70513251e+00 -2.89869993e-01]
     [ 2.50498101e-02  1.19992510e+03 -2.89869993e-01  9.70513251e+00]
     [ 9.70513251e+00 -2.89869993e-01  6.09724993e+00  5.59749981e+00]
     [-2.89869993e-01  9.70513251e+00  5.59749981e+00  6.09724993e+00]]
    

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
    plt.ylim([0, 30])
    plt.ylabel('Velocity')



plot_x()
```


![png](output_62_0.png)



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


![png](output_63_0.png)



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


![png](output_64_0.png)



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


![png](output_65_0.png)



```python
vx
```




    20




```python
dxt
```




    [16.18772751665537,
     16.63567529412575,
     16.147374730226783,
     16.281874981625705,
     15.928713287690911,
     17.46692229760085,
     17.40103497284401,
     17.062026421713288,
     17.907005778550317,
     18.375254720641937,
     18.181132130806997,
     17.49440272685743,
     17.252694128760744,
     16.92089436316335,
     17.710327283055136,
     18.8217789071423,
     18.854147184950524,
     18.793894170507553,
     19.165984167708466,
     18.175056420080956,
     18.530985413239833,
     18.46932789727426,
     18.360673805450144,
     19.076860964231305,
     19.40389128790282,
     19.14624368105296,
     19.142920465035104,
     18.86568674704688,
     18.93348722428831,
     19.265318548942048,
     19.456513427670156,
     19.666983732213474,
     19.837834907314225,
     19.94092277525696,
     19.129134886706723,
     18.947883817188085,
     18.709878192358083,
     18.796000639267685,
     18.531549936079237,
     19.250250428136653,
     19.74684205289297,
     20.184886827707576,
     19.907173798948044,
     19.78364430468734,
     19.51596986224592,
     19.31508554912709,
     19.09801537306474,
     18.51777817558998,
     18.186763787106003,
     17.363077721402533,
     17.564639011827783,
     17.613963979561852,
     18.108151284961355,
     17.639868303489955,
     18.431727796430085,
     18.296074373461714,
     18.698527838702425,
     18.927228081324802,
     18.656269686866967,
     18.31411672204309,
     18.90648542385446,
     18.900873633781295,
     18.072630553125265,
     17.454981550593473,
     17.24997312580207,
     17.342765785370275,
     17.725867603480367,
     18.5314841170311,
     18.989426824537368,
     18.745298897587666,
     18.9648217433223,
     19.455707475106216,
     19.933826575630732,
     20.100292133282895,
     19.68532078171556,
     19.368963784771918,
     20.037547846920102,
     20.293195451088334,
     19.910090069209637,
     19.614949084302125,
     19.64570678433647,
     19.569579951995035,
     19.216938823180783,
     19.95516296424816,
     20.02046544118463,
     19.06333258581755,
     18.84196979859142,
     19.232678316307062,
     18.7334112805779,
     19.520148747851692,
     19.758498985502737,
     19.895441519680798,
     20.373788600908647,
     20.865757604456412,
     21.1563933146299,
     21.14954979634911,
     21.190476037602732,
     20.796692795319778,
     20.819552567621763,
     20.84374843562099,
     20.724848163574745,
     20.070999249741703,
     19.5281010930465,
     19.16963867430607,
     19.1030477352102,
     19.520735928841617,
     19.796132840429422,
     19.72323746155313,
     19.837730897149612,
     20.24259709616823,
     20.277953088558636,
     19.792817592601363,
     19.32217991392328,
     20.125923722064254,
     20.045855769051414,
     19.77687588262529,
     19.904556051986148,
     19.923599476770097,
     19.86748695447246,
     19.108976333828423,
     19.19767541652933,
     19.454029504841518,
     19.882325422480925,
     20.6299346843047,
     20.88264874354766,
     20.443312673631446,
     20.1171674439229,
     19.9674220928632,
     20.301840370306028,
     19.66841730772011,
     18.97616110633908,
     19.563620425396266,
     19.26084070603238,
     19.520453967596424,
     19.4861727180408,
     19.570899060596698,
     19.840459610067096,
     19.160558917591946,
     19.816299172867083,
     19.674201465159978,
     19.98734630202795,
     20.018999533995963,
     19.61876502160683,
     19.847227692492527,
     19.297164500449824,
     19.499431086163145,
     19.1436302139081,
     19.30285838433413,
     19.144721520119003,
     18.862124169733562,
     19.213338646956814,
     19.097305475081928,
     18.707369354222266,
     18.738722429657994,
     18.614073707458683,
     18.66973876859177,
     18.158960985010445,
     17.96290818876917,
     18.868723715113028,
     19.463504753928287,
     19.45005719904928,
     18.961280535249006,
     19.013601793811127,
     19.083097801274054,
     18.56762534695305,
     19.056173196835548,
     18.652588482229326,
     19.035497804952747,
     19.009272305559815,
     18.630286497771177,
     18.702798129905776,
     18.976839960709587,
     18.926045303924443,
     19.747296886360374,
     19.583335104981167,
     20.6137739521043,
     20.29003089062562,
     19.414058674673246,
     19.408470028560444,
     19.630461577208845,
     19.409502893346477,
     19.73950136807978,
     19.49486748303269,
     18.95832361954169,
     19.026758579026374,
     19.45277356601995,
     19.696194203467194,
     19.663255796342927,
     19.6579491661651,
     19.078038457914356,
     19.782748706267434,
     19.495484649525668,
     19.401115032534396,
     19.222423901228325,
     19.993567671792427,
     20.139289091595575,
     19.38564047436628,
     19.205883898357254,
     19.217695190340315,
     19.622562844909137]




```python
vy
```




    10




```python
dyt
```




    [13.55488799710965,
     11.758857249809976,
     12.057197258000196,
     11.701825089572058,
     12.02355355853046,
     12.733541606817711,
     12.494555192384428,
     12.589574641868424,
     11.395931515428119,
     10.977899604990174,
     10.88348595665294,
     10.582083954561758,
     10.708515996718257,
     9.984247156783267,
     10.284386408381737,
     11.02366061378014,
     10.271916977291985,
     10.505580385621593,
     10.550639629781125,
     9.25645531759272,
     9.945342234821158,
     9.528754023196234,
     9.382639629046638,
     10.384107193861201,
     10.353980196359036,
     9.714198970939053,
     9.781180364290433,
     9.776218757621093,
     10.1518731531492,
     10.321054566672268,
     10.763815146392457,
     11.156406024684136,
     11.0240888829512,
     10.935832294312643,
     9.955867952221976,
     10.170923868983294,
     9.798557877929197,
     9.770468946535916,
     9.910223606640697,
     10.480647649916017,
     10.710697635089724,
     10.797031172452787,
     10.602181917113628,
     10.500522664870841,
     9.99917404155616,
     9.8568938280683,
     9.757415585369573,
     9.23939390753279,
     8.660443305009677,
     7.755172394705549,
     7.843624324388844,
     8.183021928513519,
     8.793823612594943,
     7.89391789027847,
     8.473594366875707,
     8.357093792661958,
     8.8469017578258,
     9.211008213298319,
     8.881692659006163,
     8.54669306139487,
     9.16012657674495,
     9.151409693599968,
     8.42944670012164,
     7.7124875764887495,
     7.320992199872961,
     7.429320220979977,
     7.740147473613558,
     8.533490015795794,
     8.972439667020508,
     8.784776144831254,
     8.898397266321334,
     9.408430611804349,
     9.999472481150073,
     10.127007534912828,
     9.946386038233069,
     9.584400046528305,
     10.246392096037697,
     10.641386409995068,
     10.270461970323096,
     10.161385942223472,
     10.12634232939573,
     9.984851548524691,
     9.680382845589445,
     10.18958329200849,
     10.113354885007318,
     9.377661022036678,
     9.271717348567531,
     9.665608753154082,
     9.043813979184323,
     9.93902265678901,
     10.149727075395937,
     10.319190600551343,
     10.862957581856202,
     11.348570263868865,
     11.544162754119538,
     11.50111910269773,
     11.503167734233257,
     11.149278787638623,
     11.307340724615981,
     11.20612916330207,
     11.104746503611207,
     10.414057853100827,
     9.872411979914457,
     9.473524022885217,
     9.417417566432826,
     9.88381764712186,
     10.134300264248713,
     10.032306408536975,
     10.059695682258953,
     10.374694063572832,
     10.420303541690812,
     9.923622342006766,
     9.522884422161775,
     10.381291187432545,
     10.306467002541028,
     10.038964038323744,
     10.124298650253355,
     10.128844281650586,
     10.061132653872326,
     9.307905286860494,
     9.416706848410913,
     9.609695418271803,
     10.15934827537505,
     10.92852656907533,
     11.189723664955169,
     10.743997485114168,
     10.320857905866205,
     10.225844113566978,
     10.589342965001574,
     9.985945408330286,
     9.285495485666136,
     9.789175534417808,
     9.480888360769937,
     9.696559628012544,
     9.659085401569195,
     9.61868057522566,
     9.797243652254423,
     9.100046673909455,
     9.789129741896112,
     9.5693376696327,
     9.856004046715812,
     9.844588011508154,
     9.435714699483594,
     9.67261031035979,
     9.080754780291322,
     9.258547279783102,
     8.952408161492025,
     9.053473535293568,
     8.844725037349981,
     8.499727309250515,
     8.83099167748749,
     8.634150472019714,
     8.330447982785154,
     8.482024176391857,
     8.421048904642536,
     8.47809650864991,
     8.03446007129115,
     7.815397995316169,
     8.740189823389036,
     9.35822905738248,
     9.377985764326844,
     8.85327288038443,
     8.841211829297045,
     8.914443413224461,
     8.435545797847224,
     8.928170894535771,
     8.515165605715083,
     8.915841895863883,
     8.873589134972029,
     8.589918890995724,
     8.744052520780341,
     9.0556581274247,
     8.966441323615932,
     9.773696317005346,
     9.635886411699198,
     10.588966047750338,
     10.283840510335098,
     9.420536961007384,
     9.416419698666907,
     9.642630761812406,
     9.435369737117679,
     9.812278236072185,
     9.536673694084561,
     8.956255911732786,
     9.03487034445616,
     9.479639621867483,
     9.712166703966513,
     9.681438728283192,
     9.702270566210775,
     9.18554248479721,
     9.895960955841927,
     9.574535661628639,
     9.483304111561756,
     9.295716782428697,
     10.098251032520709,
     10.294658383464029,
     9.577318993899503,
     9.432518408701778,
     9.445794523053102,
     9.817074440060667]



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


![png](output_73_0.png)



```python
fig = plt.figure(figsize=(16,16))
plt.scatter(xt,yt, s=20, label='State', c='k')
plt.scatter(xt[0],yt[0], s=100, label='Start', c='g')
plt.scatter(xt[-1],yt[-1], s=100, label='Goal', c='r')
plt.scatter(o[0][0],o[0][1], s=100, label='Start', c='y')
plt.scatter(o[-1][0] ,o[-1][1], s=100, label='Goal', c='r')
for i in range (len(o)):
    plt.scatter(o[i][0],o[i][1], s=20, label='State', c='b')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.legend(loc='best')
plt.axis('equal')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-63-04b1b128361f> in <module>()
         11 plt.ylabel('Y')
         12 plt.title('Position')
    ---> 13 plt.legend(loc='best')
         14 plt.axis('equal')
    

    ~\A\envs\CarND\lib\site-packages\matplotlib\pyplot.py in legend(*args, **kwargs)
       3742 @docstring.copy_dedent(Axes.legend)
       3743 def legend(*args, **kwargs):
    -> 3744     ret = gca().legend(*args, **kwargs)
       3745     return ret
       3746 
    

    ~\A\envs\CarND\lib\site-packages\matplotlib\axes\_axes.py in legend(self, *args, **kwargs)
        495                 [self],
        496                 *args,
    --> 497                 **kwargs)
        498         if len(extra_args):
        499             raise TypeError('legend only accepts two non-keyword arguments')
    

    ~\A\envs\CarND\lib\site-packages\matplotlib\legend.py in _parse_legend_args(axs, *args, **kwargs)
       1400     # No arguments - automatically detect labels and handles.
       1401     elif len(args) == 0:
    -> 1402         handles, labels = _get_legend_handles_labels(axs, handlers)
       1403         if not handles:
       1404             log.warning('No handles with labels found to put in legend.')
    

    ~\A\envs\CarND\lib\site-packages\matplotlib\legend.py in _get_legend_handles_labels(axs, legend_handler_map)
       1358         if (label
       1359                 and not label.startswith('_')
    -> 1360                 and not _in_handles(handle, label)):
       1361             handles.append(handle)
       1362             labels.append(label)
    

    ~\A\envs\CarND\lib\site-packages\matplotlib\legend.py in _in_handles(h, l)
       1342                 pass
       1343             try:
    -> 1344                 if f_h.get_facecolor() != h.get_facecolor():
       1345                     continue
       1346             except AttributeError:
    

    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()



![png](output_74_1.png)



```python
()
```




    ()

Source : https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb?create=1

