import os
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime
from .semantic_trajectory import SemanticTrajectory
import pandas as pd
from exif import Image
import csv
from PIL import Image as im


class TrajectoryReconstruction:
  def __init__(self, directory_name, drone, mission, flight, extract_to_csv=False):
    """
        Reconstruct Trajectory from geo-tagged photos metadata taken by flying object (e.g. drone)

        Parameters
        ----------
        direcotry_name : string
          The name of the folder that contains the images
        drone : string
          The type of the drone used in the flight
        mission: string
          The mission name and details
        flight: string
          The name/number of the flight
        extract_to_csv : boolean
          Determines if the extracted metadata will be saved in a csv file
    

        Examples
        --------
        Creating a trajectory from scratch:

        >>> import pandas as pd
        >>> import geopandas as gpd
        >>> import movingpandas as mpd
        >>> from fiona.crs import from_epsg

        >>> obj = mpd.TrajectoryReconstruction("inspire_2", drone="Phantom 4 Pro", mission="Petrified Forest", flight="Flight 001", extract_to_csv=True)
        >>> traj = obj.extract_metadata()
    """
    self.directory_name = directory_name
    self.drone = drone
    self.mission = mission
    self.flight = flight
    path = os.path.join("photosets", directory_name)
    self.directory_path = os.path.abspath(path)
    self.extract_to_csv = extract_to_csv
    self.trajectory_data = []


  def get_images(self):
    """
      Returns a list with the filenames of images in the given data data

      Returns
      -------
      geo_img_names : list
        filenames of the data set
    """
    geo_img_names = []
    if os.path.isdir(self.directory_path):
      for _, _, files in os.walk(self.directory_path):
        for name in files:
          if name.lower().endswith(('.png', '.jpg', '.jpeg')):
            geo_img_names.append(name)
      return geo_img_names
    raise FileNotFoundError("No directory with this name found in photosets directory")
 


  def calculate_coordinates(self, coords, ref):
    """
      Returns the coordinate in the DD form  

        Parameters
        ----------
        coords :
            The coordinate (latitude or longitude) in the DMS form
        kwargs :
            cardinal points (S or W)

        Returns
        -------
        The coordinate in the DD form
    """
    coordinates = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
      coordinates = -coordinates
    return coordinates


  def create_csv(self, data_set, data):
    """
      Creates a csv file with the extracted metadata from the given dataset 

        Parameters
        ----------
        data_set :
            The folder name of the data set
        kwargs :
            A list with the extracted metadata
    """
    header = ['X', 'Y', 'fid', 'id', 'sequence', 'trajectory_id', 'tracker', 't', 'alt', 'title', 'storage_path', 'size', 'format', 'camera_model']
    
    filename = "csv_files/" + data_set.split('//')[-1].replace(" ", "_") + '.csv'
    with open(filename, 'w', encoding='UTF8') as f:
      writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
      # write the header
      writer.writerow(header)
      # write the data
      writer.writerows(data)


  def extract_metadata(self):
    """
      Extract the metadata from the images of the data set
      and returns a new Trajectory object

      Returns
      -------
      traj : Trajectory
        the reconstructed Trajectory from the images metadata
    """
    # Get the image names
    try:
      geo_img_names = self.get_images()
    except FileNotFoundError:
      print("No directory with this name found in photosets directory")
      print("Please create a new folder called \'photosets\' and put the folder of the images in it.")
      return 0

    traj_data = []
    # Read each photo's exif info
    for file in geo_img_names:
      path = os.path.join(self.directory_path, file)
      with open(path, 'rb') as src:
        img = Image(src)
    
      latitude_coords = self.calculate_coordinates(img.gps_latitude, img.gps_latitude_ref)
      longitude_coords = self.calculate_coordinates(img.gps_longitude, img.gps_longitude_ref)
      date_time_str = img.datetime
      date_time_obj = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S')
      point_dict = {}

      point_dict['geometry'] = Point(longitude_coords, latitude_coords)
      point_dict['t'] = date_time_obj
      point_dict['alt'] = img.gps_altitude
      point_dict['trajectory_id'] = 1
      point_dict['title'] = file
      point_dict['storage_path'] = path
      point_dict['size'] = round(os.stat(path).st_size/(1024 * 1024),2)
      point_dict['format'] = file.split('.')[-1]
      point_dict['camera_model'] = img.make + "_" + img.model
      traj_data.append(point_dict)
    
    self.resize_images(geo_img_names)

    if self.extract_to_csv:
      data_to_csv = []
      for index, point in enumerate(traj_data):
        data_to_csv.append((point['geometry'].x, point['geometry'].y, index+1, index+1, index+1, point['trajectory_id'], 1, point['t'], point['alt'], point['title'], point['storage_path'], point['size'], point['format'], point['camera_model']))

      self.create_csv(self.directory_name, data_to_csv)
    

    df = pd.DataFrame(traj_data).set_index('t')
    geo_df = GeoDataFrame(df, crs='epsg:4326')
    traj = SemanticTrajectory(geo_df, 1, self.drone, self.mission, self.flight, self.directory_name)
    print("The trajectory reconstruction process has been completed successfully")
    return traj


  def resize_images(self, geo_img_names):
    """
      Creates a new directory with the scaled (resized) images of the orginal photoset to improve
      the process of trajectory visualization. The new scaled photoset is stored in a new 
      folder called 'scaled'

      Returns
      -------
    """
    # File system creation
    cwd = os.getcwd()
    os.chdir('photosets')
    if not os.path.isdir("scaled"):
      os.mkdir("scaled")
    os.chdir('scaled')
    cwd = os.getcwd()
    if not os.path.isdir(self.directory_name):
      os.mkdir(self.directory_name)
    dest_dir_name = os.path.join(cwd, self.directory_name)
    os.chdir(self.directory_name)
    cwd = os.getcwd()

    for file in geo_img_names:
      path = os.path.join(self.directory_path, file)
      with open(path, 'rb') as src:
        img = im.open(src)
        img.thumbnail((250, 250))
        
        dest_dir_name = os.path.join(cwd, file)
        img.save(dest_dir_name)
    os.chdir('../../../')
