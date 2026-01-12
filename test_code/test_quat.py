import numpy as np
import quaternion

q1 = quaternion.quaternion(0.717801992001589, 0.142621934300541, 0.166360199707428, -0.660865300709779)

angle_rad = np.radians(20)
rotation_axis = np.array([0,0,1])
rotation_q = quaternion.from_rotation_vector(rotation_axis * angle_rad)


rotated_q = rotation_q * q1
print(rotated_q)
