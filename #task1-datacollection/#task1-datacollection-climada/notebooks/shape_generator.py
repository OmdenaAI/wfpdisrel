from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString, Polygon
import argparse
import os.path
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import fiona
import re


"""
Usage: shape_generator.py [-h] -f FILENAME -n NAME -y YEAR -w WINDSPEED
                          [-b ISBUFFERED] [-r BUFFEREDRADIUS]
                          [-g GENERATESHAPEFILE] [-d DEBUG]

The options in the square brackets are optional
-f /test/test.csv
-n IVAN
-y 2004
-w max_50
-b 0 for centerpath and 1 for buffered
-r buffered radius
-g True/False for generating the shape files
-d True/False to display intermediate results

eg: python3 shape_generator.py -f /test/test.csv -n IVAN -y 2004 -w max_50 -b 1 -r 150 -g True

generates both the shapefiles and maskedfile for buffered result with radius 150 and prints the total population
"""

def decide_target_loc(source_path):
    current_path = os.getcwd()
    dest_path = os.path.abspath(os.path.join(current_path, "../data"))
    if os.path.exists(dest_path):
        return dest_path
    dest_path = os.path.split(source_path)[0]
    dest_path = os.path.join(dest_path, 'shape_files')
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    return dest_path

def evaluate_input(filename, name, year, windspeed):
    ## Evaluate the existance of the file
    if os.path.isfile(filename):
        try:
            data = pd.read_csv(filename)
            columns_list = data.columns.to_list()
            req_set = set(['name', 'year', 'lat', 'long', windspeed])
            if req_set.issubset(set(columns_list)):
                pass
            else:
                raise Shape_Generator_Exception('Unable to some or all of {}, {}, {},{},{}'.format('name', 'year', 'lat', 'long', windspeed))
            # if windspeed not in data.columns.to_list():
            #     raise Shape_Generator_Exception('Unable to find the wind speed column')
            if ((data['name'] == name).any()) and ((data[data['name'] == name]['year'] == year).any()):
                return 1
            else:
                raise Shape_Generator_Exception("Unable to find matching entries for {} name and {} year".format(name, year))
        except IOError as e:
            raise Shape_Generator_Exception('Unable to read the CSV file')
    else:
        raise Shape_Generator_Exception('Unable to find the file')


# def population_calculator(custom_df,filename, cyclone_year,extension,debug=False):
#     """
#     Generates the masked tif file and returns the sum
#     :custom_df :: Geopandas dataframe from the shape_generator class
#     :filename :: Full filename with extension
#     :cyclone_year :: Year of the cyclone
#     :extension :: extension created from the shape_generator/custom label for masked file
#     :debug :: To view intermediate results
#     :returns :: Total population
#     """


#     # geoms = custom_df.geometry.values
#     # if debug:
#     #     print(custom_df.head())
#     #     print(geoms)

#     ##  have a list of the years of available tiff file
#     ## Find the nearest year to the cyclone_year
#     ## 2010.tif, 2000.tif

#     search_path = os.path.split(filename)[0]
#     dest_path = decide_target_loc(filename)
#     try:
#         if search_path == '':
#             search_path = "./"
#         source_tif_files = [os.path.split(files)[1] for files in os.listdir(search_path) if re.match(r'.*_[0-9][0-9][0-9][0-9]_30_sec.tif',files)]
#         source_tif_files_years = [int(re.findall('\d+',x)[-2]) for x in source_tif_files]
#     except:
#         print("Error processing the files in the directory")
#         exit(1)
#     if len(source_tif_files) < 1:
#         print("Unable to find the source tif files")
#         exit(1)
#     difference_map = [abs(cyclone_year - year) for year in source_tif_files_years]
#     selected_tif_file = difference_map.index(min(difference_map))
#     selected_tif_file = "{}".format(source_tif_files[selected_tif_file])
#     selected_tif_file = os.path.join(search_path, selected_tif_file)

#     # Code for creating masked files
#     # # load the raster, mask it by the polygon and crop it
#     # with rasterio.open(selected_tif_file) as src:
#     #     out_image, out_transform = mask(src, geoms, crop=True)
#     #     out_meta = src.meta.copy()
#     # if debug:
#     #     print(out_meta)
#     # # save the resulting raster  
#     # out_meta.update({"driver": "GTiff",
#     #     "height": out_image.shape[1],
#     #     "width": out_image.shape[2],
#     #     "transform": out_transform})
    
#     # #Creates cropped as a new tif file
#     # maskedfilename = os.path.join(dest_path, "{}_masked.tif".format(extension))
#     # with rasterio.open(maskedfilename, "w", **out_meta) as dest:
#     #     dest.write(out_image)
#     with rasterio.open(selected_tif_file) as sourcetif:
#         raster_file_crs = str(sourcetif.crs).lower()
#     stats = ['min', 'max', 'mean', 'sum']
#     shape_file_crs = custom_df.crs
#     if debug:
#         print('shape files crs is {}'.format(shape_file_crs))
#         print('source tif file crs is {}'.format(raster_file_crs))
#     assert shape_file_crs == raster_file_crs
#     result = zonal_stats(custom_df, selected_tif_file, stats = stats)
#     if debug:
#         print('result is {}'.format(result))
#     population = result[0]['sum']
#     return population


class Shape_Generator_Exception(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            self.message = args[0]
        else:
            self.message = None
    
    def __str__(self):
        if self.message:
            return 'Shape Generator Exception, {}'.format(self.message)
        else:
            return 'Shape Generator Exception has been raised'
        

class Shape_Generator:
    def __init__(self, file_path, cyclone_name, cyclone_year, wind_speed, debug=False):
        """
        Initiate shape generator with the input
        :file_path : path of the filename
        :cyclone_name: name of the cyclone
        :wind_speed: Wind speed column name
        """
        evaluate_input(file_path, cyclone_name, cyclone_year, wind_speed)
        self.dest_path = decide_target_loc(file_path)
        data_df = pd.read_csv(file_path)
        cyclones = data_df[['name',
                                 'year', 'lat', 'long', wind_speed]]
        cyclones_filter = cyclones[(cyclones.year == cyclone_year) & (
                            cyclones.name == cyclone_name)]
        self.cyclone_df = pd.DataFrame(cyclones_filter)
        self.cyclone_name = cyclone_name
        self.cyclone_year = cyclone_year
        self.wind_speed = wind_speed
        self.filename = file_path
        # self.dest_path = os.path.split(file_path)[0]
        self.debug = debug
        self.extension = f"{self.cyclone_name}_{self.cyclone_year}_{self.wind_speed}_"
        if self.debug:
            print("Dataframe after filtering ", self.cyclone_df.head())
        self.cyclone_gdf = self.__generate_geopandas_df__(self.cyclone_df)
        self.final_gdf = pd.DataFrame()
        pass

    @staticmethod
    def __generate_geopandas_df__(df):
        """
        Generate the geopandas dataframe using latitude and longitude
        :df: pandas dataframe with columns lat and long
        :return: Geopandas dataframe
        """
        geometry = [Point(xy) for xy in zip(df.long, df.lat)]
        return GeoDataFrame(df, geometry=geometry)
    
    def generate_center_path(self, generateFile=False):
        """
        Generate the output files for centerpath
        :generateFile : True/False to generate the files using the computed centerpath
        """
        group_gdf = self.cyclone_gdf.groupby(['name', 'year'])[
            'geometry'].apply(lambda x: LineString(x.tolist()))
        if self.debug:
            print(f'Length of linestring dataframe {len(group_gdf)}')
            print(f'First five row of {group_gdf.head()}')
        final_gdf = geopandas.GeoDataFrame(group_gdf, geometry='geometry')
        if self.debug:
            print(final_gdf)
            final_gdf.plot()
            if generateFile:
                plt.show()
            else:
                plt.show(block=False)
        if (final_gdf.crs == None):
            final_gdf.crs = "epsg:4326"
        self.extension += "centerpath"
        self.final_gdf = final_gdf
        if generateFile:
            self.generate_shapefile()
        return self.final_gdf

    def generate_shapefile(self):
        """
        Generate the output files from the geopandas dataframe
        :extension: extension of the output file
        """
        filename = self.extension + ".shp"
        if (self.dest_path != ''):
            outfile = os.path.join(self.dest_path, filename)
        else:
            outfile = filename
        if self.debug:
            print(f"outfile is {outfile}")
        self.final_gdf.to_file(filename=outfile, driver="ESRI Shapefile")
        print("Successfully created Shape files in {}".format(self.dest_path))
    

    def generate_buffered_file(self, radius=0, generateFile=False):
        """
        Generate the buffered output
        :radius The required radius for the buffered output
        """
        # generate the centerpath first and make it buffered
        self.generate_center_path(generateFile=False)
        self.extension = '_'.join(self.extension.split('_')[:-1])
        buffer_radius = radius
        # If no radius is mentioned
        if radius == 0:
            # compute mean
            buffer_radius = round(self.cyclone_df[self.wind_speed].mean(),2)
            # compute median
            buffer_radius = round(self.cyclone_df[self.wind_speed].median(),2)
            if self.debug:
                print(f'buffered radius = {buffer_radius}')
        # Project to crs that uses meters as distance measure
        cyclone_3395 = self.final_gdf.to_crs('epsg:3395')
        cyclone_3395['geometry'] = cyclone_3395.buffer(buffer_radius*1000)
        buffered_cyclone_gdf = cyclone_3395.to_crs('epsg:4326')
        if self.debug:
            print(buffered_cyclone_gdf.head())
            buffered_cyclone_gdf.plot()
            plt.show()
        self.final_gdf = buffered_cyclone_gdf
        self.extension += f"_{buffer_radius}_buffered"
        if generateFile:
            self.generate_shapefile()
        return self.final_gdf


class Population_Calculator(Shape_Generator):
    def __init__(self, file_path, cyclone_name, cyclone_year, wind_speed, debug=False):
        super().__init__(file_path, cyclone_name, cyclone_year, wind_speed, debug=debug)

    def get_population(self):

        if self.final_gdf.empty:
            self.generate_center_path(generateFile=False)

        # geoms = custom_df.geometry.values
        # if debug:
        #     print(custom_df.head())
        #     print(geoms)

        ##  have a list of the years of available tiff file
        ## Find the nearest year to the cyclone_year
        ## 2010.tif, 2000.tif
        search_path = os.path.split(self.filename)[0]
        if os.path.exists(os.path.join(search_path, 'Gridded Population of the World(GPW)')):
            search_path = os.path.join(search_path, 'Gridded Population of the World(GPW)')
        try:
            if search_path == '':
                search_path = "./"
            source_tif_files = [os.path.split(files)[1] for files in os.listdir(search_path) if re.match(r'.*_[0-9][0-9][0-9][0-9]_30_sec.tif',files)]
            source_tif_files_years = [int(re.findall('\d+',x)[-2]) for x in source_tif_files]
        except:
            print("Error processing the files in the directory")
            exit(1)
        if len(source_tif_files) < 1:
            print("Unable to find the source tif files")
            exit(1)
        difference_map = [abs(self.cyclone_year - year) for year in source_tif_files_years]
        selected_tif_file = difference_map.index(min(difference_map))
        selected_tif_file = "{}".format(source_tif_files[selected_tif_file])
        selected_tif_file = os.path.join(search_path, selected_tif_file)

        # Code for creating masked files
        # # load the raster, mask it by the polygon and crop it
        # with rasterio.open(selected_tif_file) as src:
        #     out_image, out_transform = mask(src, geoms, crop=True)
        #     out_meta = src.meta.copy()
        # if debug:
        #     print(out_meta)
        # # save the resulting raster  
        # out_meta.update({"driver": "GTiff",
        #     "height": out_image.shape[1],
        #     "width": out_image.shape[2],
        #     "transform": out_transform})
    
        # #Creates cropped as a new tif file
        # maskedfilename = os.path.join(dest_path, "{}_masked.tif".format(extension))
        # with rasterio.open(maskedfilename, "w", **out_meta) as dest:
        #     dest.write(out_image)
        with rasterio.open(selected_tif_file) as sourcetif:
            raster_file_crs = str(sourcetif.crs).lower()
            stats = ['min', 'max', 'mean', 'sum']
            shape_file_crs = self.final_gdf.crs
            if self.debug:
                print('shape files crs is {}'.format(shape_file_crs))
                print('source tif file crs is {}'.format(raster_file_crs))
            assert shape_file_crs == raster_file_crs
        result = zonal_stats(self.final_gdf, selected_tif_file, stats = stats)
        if self.debug:
            print('result is {}'.format(result))
        population = result[0]['sum']
        return population


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a shape file')
    parser.add_argument("-f", "--filename", dest = "filename", help="Input datafile",required=True)
    parser.add_argument("-n", "--name", dest = "name", help="Cyclone name", required=True)
    parser.add_argument("-y", "--year",dest ="year", help="Cyclone year", type=int, required=True)
    parser.add_argument("-w", "--windspeed",dest = "windspeed", help="Wind Speed column name", required=True)
    parser.add_argument("-b", "--isbuffered", dest="isbuffered", help="Centerpath : 0 / buffered : 1", type=int, default=0)
    parser.add_argument("-r", "--bufferedradius", dest="bufferedradius", help="buffered radius", default=0, type=int)
    parser.add_argument("-g", "--generateshapefile", dest="generateshapefile", help="generate shape file", default=False, type=bool)
    parser.add_argument('-d', "--debug", dest="debug", help="Display the dataframes and plots", default=False)

    args = parser.parse_args()
    # shape_object = Shape_Generator(args.filename, args.name, args.year, args.windspeed,args.debug)
    # output = ''
    # if args.isbuffered:
    #     output = shape_object.generate_buffered_file(radius = args.bufferedradius,generateFile=args.generateshapefile)
    # else:
    #     output = shape_object.generate_center_path(generateFile=args.generateshapefile)
    # print(population_calculator(output, args.filename, args.year, shape_object.extension))
    pop_object = Population_Calculator(args.filename, args.name, args.year, args.windspeed, args.debug)
    output = ''
    if args.isbuffered:
        output = pop_object.generate_buffered_file(radius=args.bufferedradius, generateFile=args.generateshapefile)
    else:
        output = pop_object.generate_center_path(generateFile=args.generateshapefile)
    print(pop_object.get_population())