import numpy as np


def wm(data, err):
	weight_sum = 0
	data_sum = 0
	for i in range(len(data)):
		weight = 1/ (err[i]**2)
		data_sum += data[i] * weight
		weight_sum += weight

	total_err = 1 / weight_sum * np.sqrt(sum((1/(y**2))**2 * y**2 for y in err))
	return data_sum / weight_sum, total_err


debris = [2.55, 2.55, 2.57]
droplet = [ 4.81, 4.89, 4.75]
defect = [ 3.08, 3.16, 3.24]

debris_err = [ 0.01, 0.01, 0.01]
droplet_err = [ 0.08, 0.22, 0.23]
defect_err = [ 0.10, 0.11, 0.11]

print('EXPONENT')
print('VN')
print(wm(debris, debris_err))
print(wm(droplet, droplet_err))
print(wm(defect, defect_err))

debris = [2.52, 2.57, 2.60]
droplet = [4.34, 4.32, 4.37]
defect = [2.88, 2.82, 2.77]

debris_err = [0.01, 0.01, 0.01]
droplet_err = [0.05, 0.06, 0.07]
defect_err = [0.13, 0.09, 0.10]

print('Moore')
print(wm(debris, debris_err))
print(wm(droplet, droplet_err))
print(wm(defect, defect_err))



def process(data, err):
	out_data = np.exp(data)
	out_err = []

	for i in range(len(err)):
		out_err.append(out_data[i] * data[i] * err[i] / np.exp(1))
	print(out_data, out_err)
	return out_data, out_err


# VN const
debris = [-3.985509, -4.001598, -4.045017]
droplet = [-7.581093, -7.798826, -7.420146]
defect = [-3.883145, -4.053539, -4.251902]

debris_err = [0.02, 0.02, 0.01]
droplet_err = [0.20, 0.57, 0.59]
defect_err = [0.23, 0.29, 0.29]



print('CONSTANT')

print('VN')
print(wm(debris, debris_err))
print(wm(droplet, droplet_err))
print(wm(defect, defect_err))



# Moore const
debris = [-5.021302 , -5.143954 , -5.232595 ]
droplet = [-8.312495 , -8.246143 , -8.382914 ]
defect = [-4.760676 , -4.590816 , -4.450389 ]

debris_err = [0.03, 0.02, 0.01]
droplet_err = [.16, 0.16, 0.20]
defect_err = [0.34, 0.25, 0.26]



print('Moore')
print(wm(debris, debris_err))
print(wm(droplet, droplet_err))
print(wm(defect, defect_err))


