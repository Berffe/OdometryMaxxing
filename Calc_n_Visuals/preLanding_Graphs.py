import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve 

def main():
	z0 = 1 ; paw = 0.1
	flux = -1

	z = lambda t : z0*np.exp(flux*t) 
	z_paw = lambda t : z0*np.exp(flux*t) - paw
	v_z = lambda t : flux*z0*np.exp(flux*t)
	t = np.linspace(0, 10, 100)
	# t_land = 1/flux*np.log(paw/z0)

	omega0 = 1 
	taus = [0.1, 0.5, 1, 2]
	omega = lambda t, tau : omega0*(1 - tau/(t + tau))
	height = lambda t, tau : z0*np.exp(omega0*t)*(t/tau + 1)**(-omega0*t) - paw
	vertical_speed = lambda t, tau : omega(t, tau)*(height(t, tau)+paw)
	
	plt.figure()
	for tau in taus : 
		plt.plot(t, height(t, tau), label = f"tau : {tau}")
		t_land = fsolve(height, x0=2, args=tau)[0]
		plt.scatter(t_land, height(t_land, tau))
	plt.xlabel("Temps")
	plt.ylabel("Hauteur")
	plt.legend()
	plt.grid()

	plt.figure()
	for tau in taus : 
		plt.plot(t, vertical_speed(t, tau), label = f"tau : {tau}")
		t_land = fsolve(height, x0=2, args=tau)[0]
		plt.scatter(t_land, vertical_speed(t_land, tau))
	plt.xlabel("Temps")
	plt.ylabel("Vitesse")
	plt.legend()
	plt.grid()
	plt.show()

	# print("Speed at touchdown is : ", v_z(t_land))
	# plt.figure()
	# plt.plot(t, z(t), color = 'y', linestyle="--", label = "z(t) asymptotique")
	# plt.plot(t, z_paw(t), color = 'b', linestyle="-", label = "z'(t) avec jambes")
	# plt.plot(t, v_z(t), color = 'r', linestyle="-", label = "vitesse")
	# plt.scatter(t_land, v_z(t_land), color = 'r', label = "vitesse contact")
	# plt.scatter(t_land, z_paw(t_land), color = 'b', label = "point de contact")
	# # plt.scatter(t_land, z_paw(t_land), color = 'r')
	# plt.xlabel("Temps")
	# plt.ylabel("Vitesse/ Hauteur")
	# plt.legend()
	# plt.grid()
	# plt.show()

	# touchdown = lambda z_init, div_flux : 1/div_flux*np.log(paw/z_init)
	# z_init = np.linspace(0.1, 2, 100)
	# div_flux = [-0.1, -0.8, -1, -1.2, -1.5, -2]

	# plt.figure()
	# for f in div_flux : 
	# 	time = touchdown(z_init, f)
	# 	plt.plot(z_init, v_z(time), label = f"WT_set : {f}")
	# plt.grid()
	# plt.xlabel("Hauteur initial")
	# plt.ylabel("Vitesse relative verticale au contact")
	# plt.legend()
	# plt.show()



if __name__ == '__main__' :
	main()