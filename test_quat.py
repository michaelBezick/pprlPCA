import numpy as np
import quaternion

q1 = quaternion.quaternion(0.364705, 0.279848, 0.115917, -0.880476)

angle_rad = np.radians(50)
rotation_axis = np.array([0,0,1])
rotation_q = quaternion.from_rotation_vector(rotation_axis * angle_rad)


rotated_q = rotation_q * q1

#rotate up with x

angle_rad = -np.radians(10)
rotation_axis = np.array([1,0,0])
rotation_q = quaternion.from_rotation_vector(rotation_axis * angle_rad)

rotated_q = rotation_q * rotated_q

print(rotated_q)
