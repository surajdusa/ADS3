#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import cluster_tools
import errors


# In[2]:


pop_df = pd.read_csv('Population, total.csv')
# melt the dataframe
pop_df = pop_df.melt(id_vars = 'date', var_name = 'country', value_name = 'population')

co2_df = pd.read_csv('CO2 emissions (metric tons per capita).csv')
# melt the dataframe
co2_df = co2_df.melt(id_vars = 'date', var_name = 'country', value_name = 'co2_emission_pc')


# In[3]:


pop_df.head()


# In[4]:


co2_df.head()


# In[5]:


# Keep observations that are not missing in CO2 in population
pop_df = pop_df.loc[co2_df.co2_emission_pc.notna(), :].reset_index(drop = True)
co2_df = co2_df.loc[co2_df.co2_emission_pc.notna(), :].reset_index(drop = True)
assert co2_df.shape == pop_df.shape, "Unequal number of rows"

# Create the main dataframe with data
df = pd.DataFrame({"population":pop_df.population, 'co2_emission_pc':co2_df.co2_emission_pc})
df.head()


# In[6]:


# Compute correlation
cluster_tools.map_corr(df)


# In[7]:


# Normalize the data
df_scaled, df_min, df_max = cluster_tools.scaler(df)


# In[8]:


# Clustering the data, first find the optimal k using elbow plot
inertia = []
K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters = k, n_init = 'auto')
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'r3-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.title('Elbow Plot to Find Optimal k')
plt.grid()
plt.show()


# In[9]:


# K = 4 is the optimal value
optimal_k = 4
kmeans = KMeans(n_clusters = 4, n_init = 'auto')
kmeans.fit(df_scaled)
# Calculate the cluster centers
cluster_centers = kmeans.cluster_centers_
cluster_centers_backscaled = cluster_tools.backscale(cluster_centers, df_min, df_max)
# Add labels column to the original data
df['cluster'] = kmeans.labels_

# Visualize the cluster centres
colors = ['#f3f847', '#0e4bd2', '#d20eb9', '#0ed210']
for i in range(optimal_k):
    x = df.loc[df['cluster'] == i, 'co2_emission_pc']
    y = df.loc[df['cluster'] == i, 'population']
    plt.scatter(x, y, color = colors[i])
    plt.scatter(cluster_centers_backscaled[i, 1], cluster_centers_backscaled[i, 0], 
                marker = 'o', s = 100, color = 'black')
plt.xlabel('CO2 emissions per capita')
plt.ylabel('Population')
plt.title('Clusters of Countries')
plt.show()


# In[17]:


# Curve fitting data for cluster 3 (2 in this case)
# Estimating the average CO2 per capita per year given the yearly mean population of those countries
mask = df.cluster == 2
country_c3 = pop_df.country[mask]
date_c3 = pop_df.date[mask]
df_c3 = df.loc[mask, ['population', 'co2_emission_pc']]
df_c3['date'] = date_c3
df_c3 = df_c3.groupby('date').agg({'population':'mean', 'co2_emission_pc':'mean'})

# Visualize the behaviour of the data before fitting
print(f"Countries in Cluster 3: {country_c3.unique()}")
df_c3.plot('population', 'co2_emission_pc', kind = 'scatter')
plt.title("Third Cluster Data")
plt.ylabel('CO2 emission per capita')
plt.grid()
plt.show()

def poly_3f(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

xdata = df_c3['population'].to_numpy()
ydata = df_c3['co2_emission_pc'].to_numpy()
popt, pcov = curve_fit(poly_3f, xdata, ydata)
# Using the fit parameters
yfit = poly_3f(xdata, *popt)
# Get the sigmas
sigmas = np.sqrt(np.diag(pcov))

# Plot the original data with line of best fit
df_c3.plot('population', 'co2_emission_pc', kind = 'scatter', label = 'Actual CO2')
plt.scatter(xdata, yfit, label = "Estimated CO2", color = 'red')
if all(sigmas > 1): # Check if all sigmas are greater than 1
    ylower, yupper = errors.err_ranges(xdata, poly_3f, popt, sigmas)
    plt.fill_between(xdata, ylower, yupper, alpha = 0.2, label = 'confidence range')
plt.title("Actual and fitted data")
plt.ylabel('CO2 emission per capita')
plt.grid()
plt.legend()
plt.show()


# In[15]:


# Curve fitting data for cluster 2 (1 in this case)
# Estimating the average CO2 per capita per year given the yearly mean population of those countries
mask = (df.cluster == 1)
country_c2 = pop_df.country[mask]
date_c2 = pop_df.date[mask]
df_c2 = df.loc[mask, ['population', 'co2_emission_pc']]
df_c2['date'] = date_c2
df_c2 = df_c2.groupby('date').agg({'population':'mean', 'co2_emission_pc':'mean'})
# Visualize the behaviour of the data before fitting
print(f"Countries in Cluster 2: {country_c2.unique()}")
df_c2.plot('population', 'co2_emission_pc', kind = 'scatter')
plt.title("Second Cluster Data")
plt.ylabel('CO2 emission per capita')
plt.grid()
plt.show()

# Use the second order polynomial
def poly_2f(x, a, b, c):
    return a*x**2 + b*x + c

xdata = df_c2['population'].to_numpy()
ydata = df_c2['co2_emission_pc'].to_numpy()
popt, pcov = curve_fit(poly_2f, xdata, ydata)
# Using the fit parameters
yfit = poly_2f(xdata, *popt)
# Get the sigmas
sigmas = np.sqrt(np.diag(pcov))

# Plot the original data with line of best fit
df_c2.plot('population', 'co2_emission_pc', kind = 'scatter', label = 'Actual CO2')
plt.scatter(xdata, yfit, label = "Estimated CO2", color = 'red')
if all(sigmas > 1): # Check if all sigmas are greater than 1
    ylower, yupper = errors.err_ranges(xdata, poly_2f, popt, sigmas)
    plt.fill_between(xdata, ylower, yupper, alpha = 0.2, label = 'confidence range')
plt.title("Actual and fitted data")
plt.ylabel("CO2 emission per capita")
plt.grid()
plt.legend()
plt.show()

