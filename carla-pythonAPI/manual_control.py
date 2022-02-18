#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# https://github.com/carla-simulator/carla/issues/2666
"""
An example of client-side bounding boxes with basic car controls.

Controls:

	W			 : throttle
	S			 : brake
	AD			 : steer
	Space		 : hand-brake

	ESC			 : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

from matplotlib import image

try:
    # sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist-*%d.%d-%s.egg' % (
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import time
start = time.time()
import carla
import weakref
import random
import cv2

try:
	import pygame
	from pygame.locals import K_ESCAPE
	from pygame.locals import K_SPACE
	from pygame.locals import K_a
	from pygame.locals import K_d
	from pygame.locals import K_s
	from pygame.locals import K_w
	from pygame.locals import K_m
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# VIEW_WIDTH = 1920//4
# VIEW_HEIGHT = 1080//4

VIEW_WIDTH = 512
VIEW_HEIGHT = 512
VIEW_FOV = 90
FILE_NAME = 'kitti'
DIRECTORY = "carla_seq/11/"



# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
	"""
	Basic implementation of a synchronous client.
	"""

	def __init__(self):
		self.client = None
		self.world = None
		self.camera = None
		self.depth_camera = None
		self.semantic_camera = None
		self.car = None
		self.display = None
		self.depth_display = None
		self.image = None
		self.depth_image = None
		self.semantic_image = None
		self.capture = True
		self.depth_capture = True
		self.semantic_capture = True
		self.counter = 0
		self.depth = None
		self.semantic = None
		self.pose = []
		self.log = False
		

	def camera_blueprint(self, type = 'rgb'):
		"""
		type: rgb, depth or semantic
		Returns camera blueprint.
		"""

		camera_bp = self.world.get_blueprint_library().find('sensor.camera.' + type)
		if(camera_bp == None):
			print(type + 'camera was not created')
		camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
		camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
		camera_bp.set_attribute('fov', str(VIEW_FOV))
		
		return camera_bp


	def set_synchronous_mode(self, synchronous_mode):
		"""
		Sets synchronous mode.
		"""

		settings = self.world.get_settings()
		settings.synchronous_mode = synchronous_mode
		self.world.apply_settings(settings)

	def setup_car(self):
		"""
		Spawns actor-vehicle to be controled.
		"""

		car_bp = self.world.get_blueprint_library().filter('model3')[0]
		location = random.choice(self.world.get_map().get_spawn_points())
		self.car = self.world.spawn_actor(car_bp, location)

	def setup_camera(self):
		"""
		Spawns actor-camera to be used to render view.
		Sets calibration for client-side boxes rendering.
		"""

		# camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
		camera_transform = carla.Transform(carla.Location(x=0, z=3.2), carla.Rotation(pitch=0))
		self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
		weak_self = weakref.ref(self)
		self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.camera.calibration = calibration
		
	def setup_depth_camera(self):
		"""
		Spawns actor-camera to be used to render view.
		Sets calibration for client-side boxes rendering.
		"""

		depth_camera_transform = carla.Transform(carla.Location(x=0, z=3.2), carla.Rotation(pitch=0))
		self.depth_camera = self.world.spawn_actor(self.camera_blueprint('depth'), depth_camera_transform, attach_to=self.car)
		weak_depth_self = weakref.ref(self)
		self.depth_camera.listen(lambda depth_image: weak_depth_self().set_depth_image(weak_depth_self, depth_image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.depth_camera.calibration = calibration

	def setup_semantic_camera(self):
		"""
		Spawns actor-camera to be used to render view.
		Sets calibration for client-side boxes rendering.
		"""

		semantic_camera_transform = carla.Transform(carla.Location(x=0, z=3.2), carla.Rotation(pitch=0))
		self.semantic_camera = self.world.spawn_actor(self.camera_blueprint('semantic_segmentation'), semantic_camera_transform, attach_to=self.car)
		# self.semantic_camera.listen(lambda image: image.save_to_disk('%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette))
		weak_semantic_self = weakref.ref(self)
		self.semantic_camera.listen(lambda semantic_image: weak_semantic_self().set_semantic_image(weak_semantic_self, semantic_image))

		calibration = np.identity(3)
		calibration[0, 2] = VIEW_WIDTH / 2.0
		calibration[1, 2] = VIEW_HEIGHT / 2.0
		calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
		self.semantic_camera.calibration = calibration

	def control(self, car):
		"""
		Applies control to main car based on pygame pressed keys.
		Will return True If ESCAPE is hit, otherwise False to end main loop.
		"""

		keys = pygame.key.get_pressed()
		if keys[K_ESCAPE]:
			return True

		control = car.get_control()
		control.throttle = 0
		
		if keys[K_w]:
			control.throttle = 1
			control.reverse = False
		elif keys[K_s]:
			control.throttle = 1
			control.reverse = True
		if keys[K_a]:
			control.steer = max(-1., min(control.steer - 0.05, 0))
		elif keys[K_d]:
			control.steer = min(1., max(control.steer + 0.05, 0))
		else:
			control.steer = 0
		control.hand_brake = keys[K_SPACE]
		if keys[K_m]:
			if self.log:
				self.log = False
				np.savetxt('log/pose.txt',self.pose)
			else:
				self.log = True
			pass

		
		car.apply_control(control)
		return False

	@staticmethod
	def set_image(weak_self, img):
		"""
		Sets image coming from camera sensor.
		The self.capture flag is a mean of synchronization - once the flag is
		set, next coming image will be stored.
		"""

		self = weak_self()
		if self.capture:
			self.image = img
			self.capture = False

	@staticmethod
	def set_depth_image(weak_depth_self, depth_img):
		"""
		Sets image coming from camera sensor.
		The self.capture flag is a mean of synchronization - once the flag is
		set, next coming image will be stored.
		"""

		self = weak_depth_self()
		if self.depth_capture:
			self.depth_image = depth_img
			self.depth_capture = False

	@staticmethod
	def set_semantic_image(weak_semantic_self, semantic_img):
		"""
		Sets image coming from camera sensor.
		The self.capture flag is a mean of synchronization - once the flag is
		set, next coming image will be stored.
		"""

		self = weak_semantic_self()
		if self.semantic_capture:
			self.semantic_image = semantic_img
			self.semantic_capture = False


	def render(self, display, count):
		"""
		Transforms image from camera sensor and blits it to main pygame display.
		"""
		if self.image is not None:
			# self.image.save_to_disk("../carlaData/image/img" + str(count) + ".png")
			# self.image.save_to_disk("image_2/%06d.png" % count)
			array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (self.image.height, self.image.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			display.blit(surface, (0, 0))

	def depth_render(self, depth_display, count):	
		if self.depth_image is not None:
			# self.depth_image.save_to_disk("../carlaData/depth/depth" + str(count) + ".png")
			#i = np.array(self.depth_image.raw_data)
			i = np.array(self.depth_image)
			i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 1))
			i3 = i2[:, :, :]
			self.depth = i3
			#cv2.imshow("depth_image", self.depth)

	def semantic_render(self, semantic_display, count):
		if self.semantic_image is not None:
			# self.semantic_image.save_to_disk("../carlaData/sem/id%06d.png" % count)
			# self.semantic_image.save_to_disk("../carlaData/sem/id%06d.png" % count, carla.ColorConverter.CityScapesPalette)
			
			# self.semantic_image.save_to_disk("semantic/%06d.png" % count)
			# self.semantic_image.save_to_disk("../carlaData/sem/id%06d.png" % count, carla.ColorConverter.CityScapesPalette)
			# self.semantic_image.save_to_disk("../carlaData/sem/" + str(count) + ".png", carla.ColorConverter.CityScapesPalette)

			# self.semantic_image.convert(carla.ColorConverter.CityScapesPalette)
			# self.semantic_image.convert(carla.ColorConverter)
			i = np.array(self.semantic_image.raw_data)

			i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
			i3 = i2[:, :, :3]
			# print(i3)
			self.semantic = i3
			cv2.imshow("semantic_image", self.semantic)

	def save(self, count, video_out, file_ptr, ini, time_ptr, time):
		pos = self.car.get_transform()
		
		phi = pos.rotation.roll
		theta = pos.rotation.pitch
		psi = pos.rotation.yaw

		r11 = np.cos(phi)*np.cos(theta)
		r12 = np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(theta)*np.cos(psi)
		r13 = np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(theta)*np.cos(psi)
		tx = pos.location.x - ini[0]
		r21 = np.sin(phi)*np.cos(theta)
		r22 = np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(theta)*np.cos(psi)
		r23 = np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(theta)*np.sin(psi)
		ty = pos.location.y - ini[1]
		r31 = -np.sin(theta)
		r32 = np.cos(theta)*np.sin(psi)
		r33 = np.cos(theta)*np.cos(psi)
		tz = pos.location.z - ini[2]
		str_pos = "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n" .format(
			r11,r12,r13,tx,r31,r32,r33,tz,r21,r22,r23,ty) #TODO check z axis orientation
		
		# str_pos = "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n" .format(
		# 	r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz)

		# str_pos = "x:{:.4f}, y:{:.4f}, z:{:.4f}, roll:{:.4f}, pitch:{:.4f}, yaw:{:.4f}\n".format(pos.location.x, pos.location.y, pos.location.z, pos.rotation.roll, pos.rotation.pitch, pos.rotation.yaw)
		# str_pos = "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(pos.location.x, pos.location.y, pos.location.z, pos.rotation.roll, pos.rotation.pitch, pos.rotation.yaw)
		file_ptr.write(str_pos)

		time_str = "{:.6f}\n" .format(time)
		time_ptr.write(time_str)

		# if self.image is not None:
		# 	array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
		# 	array = np.reshape(array, (self.image.height, self.image.width, 4))
		# 	array = array[:, :, :3]
		# 	# array = array[:, :, ::-1]
		# 	video_out.write(array)
		# 	# self.image.save_to_disk("../carlaData/image/id%05d.png" % count)

	def depth_save(self, count, video_out):
		if self.depth_image is not None:
			self.depth_image = np.array(self.depth_image.raw_data)
			self.depth_image = self.depth_image.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
			self.depth_image = self.depth_image[:,:,:3]
			video_out.write(self.depth_image)
			# self.depth_image.save_to_disk("../carlaData/depth/id%05d.png" % count)
	
	def semantic_save(self, count, video_out):
		if self.semantic_image is not None:
			semantic_save = np.array(self.semantic_image.raw_data)
			semantic_save = semantic_save.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
			semantic_save = semantic_save[:,:,2]
			video_out.write(semantic_save)
			# self.semantic_image.save_to_disk("../carlaData/sem/id%05d.png" % count)
			# self.semantic_image.convert(carla.ColorConverter.CityScapesPalette)
			# self.semantic_image.save_to_disk("../carlaData/semantic/idcolor%05d.png" % count)


	def log_data(self):
			global start
			freq = 1/(time.time() - start)

	#		sys.stdout.write("\rFrequency:{}Hz		Logging:{}".format(int(freq),self.log))
			
			sys.stdout.flush()
			if self.log:
				name ='log/' + str(self.counter) + '.png'
				self.depth_image.save_to_disk(name)
				position = self.car.get_transform()
				pos=None
				pos = (int(self.counter), position.location.x, position.location.y, position.location.z, position.rotation.roll, position.rotation.pitch, position.rotation.yaw)
				self.pose.append(pos)
				self.counter += 1
			start = time.time()
		

	def game_loop(self):
		"""
		Main program loop.
		"""

		try:
			pygame.init()

			self.client = carla.Client('127.0.0.1', 2000)
			self.client.set_timeout(4.0)
			self.world = self.client.get_world()
			#self.world = self.client.load_world('Town11')
			# self.world.set_weather(carla.WeatherParameters.ClearNoon)
			# print(self.world.get_weather())

			# print(self.client.get_available_maps())
			# return
			self.setup_car()
			self.setup_camera()
			#self.setup_depth_camera()
			self.setup_semantic_camera()

			self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
			#self.depth_display = cv2.namedWindow('depth_image')
			self.semantic_display = cv2.namedWindow('semantic_image')

			pygame_clock = pygame.time.Clock()

			self.set_synchronous_mode(True)
			vehicles = self.world.get_actors().filter('vehicle.*')
			count = 0
			# Wait for the next tick and retrieve the snapshot of the tick.
			# world_snapshot = self.world.wait_for_tick()

			# # Register a callback to get called every time we receive a new snapshot.
			# self.world.on_tick(lambda world_snapshot: self.saveImages(world_snapshot))

			# fourcc_camera = cv2.VideoWriter_fourcc(*'XVID')
			# out_camera = cv2.VideoWriter('out_camera_' + FILE_NAME + '.avi', fourcc_camera, 20.0, (VIEW_WIDTH,  VIEW_HEIGHT))

			# fourcc_seg = cv2.VideoWriter_fourcc(*'XVID')
			# out_seg = cv2.VideoWriter('out_seg_' + FILE_NAME + '.avi', fourcc_seg, 20.0, (VIEW_WIDTH,  VIEW_HEIGHT),0)

			#fourcc_depth = cv2.VideoWriter_fourcc(*'XVID')
			#out_depth = cv2.VideoWriter('out_depth_kitti.avi', fourcc_depth, 20.0, (VIEW_WIDTH,  VIEW_HEIGHT))

			file_ptr = open(DIRECTORY + 'pos_' + FILE_NAME + '.txt', 'w')
			time_ptr = open(DIRECTORY + 'times.txt', 'w')

			pos = self.car.get_transform()
			ini_x = pos.location.x
			ini_y = pos.location.y
			ini_z = pos.location.z
			ini = [ini_x, ini_y, ini_z]

			begin_t = time.time()

			images = []
			semantic = []

			while True:
				self.world.tick()
				self.capture = True
				#self.depth_capture = True
				self.semantic_capture = True
				pygame_clock.tick_busy_loop(30)
				self.render(self.display, count)
				#self.depth_render(self.depth_display, count)
				self.semantic_render(self.semantic_display, count)
				images.append(self.image)
				semantic.append(self.semantic_image)
				thread_img = threading.Thread(target = saveImage, args = (self.image, count))
				thread_label = threading.Thread(target = saveLabel, args = (self.semantic_image, count))
				thread_img.start()
				thread_label.start()
				# self.image.save_to_disk("image_2/%06d.png" % count)
				# self.semantic_image.save_to_disk("semantic/%06d.png" % count)
				
				curr_t = time.time()
				self.save(count, None, file_ptr, ini, time_ptr, curr_t-begin_t)
				#self.depth_save(count, out_depth)
				# self.semantic_save(count, out_seg)
				pygame.display.flip()
				pygame.event.pump()
				self.log_data()
				cv2.waitKey(1)
				if self.control(self.car):
					# i=0
					# for image, label, in zip(images, semantic):
					# 	image.save_to_disk("image_2/%06d.png" % i)
					# 	label.save_to_disk("semantic/%06d.png" % i)
					# 	i+=1
					return

				count += 1

		#except Exception as e: print(e)
		finally:
			print(count)
			self.set_synchronous_mode(False)
			self.camera.destroy()
			#self.depth_camera.destroy()
			self.semantic_camera.destroy()
			self.car.destroy()
			pygame.quit()
			cv2.destroyAllWindows()
			file_ptr.close()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


import threading

def saveImage(image, i):
    image.save_to_disk(DIRECTORY + "image_2/%06d.png" % i)

def saveLabel(label, i):
    label.save_to_disk(DIRECTORY + "semantic/%06d.png" % i)




# thread = threading.Thread(target=f)
# thread.start()

def main():
	"""
	Initializes the client-side bounding box demo.
	"""

	try:
		client = BasicSynchronousClient()
		client.game_loop()
	finally:
		print('EXIT')


if __name__ == '__main__':
	main()