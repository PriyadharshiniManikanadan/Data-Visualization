class Univariate():
    
# quanQual Seperation 
    
    def quanQual(dataset):
        qual = []
        quan = []
        for columnName in dataset.columns:
            #print(columnName)
            if dataset[columnName].dtype == 'object':
                #print("qual")
                qual.append(columnName)
            else:
                #print("quan")
                quan.append(columnName)
        return quan,qual

    
# Central Tendency, Percentile, IQR, Skewness, Kurtosis
    
    def Univariate(dataset,quan):
        descriptive = pd.DataFrame(index=['Mean', 'Median', 'Mode', 'Q1:25%', 'Q2:50%', "Q3:75%", "99%", "Q4:100%",
                                          "IQR", "1.5-Rule", "Lesser_Outlier", "Greater_Outlier", "Min", "Max",
                                          "Skew", "Kurtosis", "Var", "StdD"], columns = quan)
        for columnName in quan:
            descriptive[columnName]['Mean'] = dataset[columnName].mean()    
            descriptive[columnName]['Median'] = dataset[columnName].median()
            descriptive[columnName]['Mode'] = dataset[columnName].mode()[0]     
            descriptive[columnName]['Q1:25%'] = dataset.describe()[columnName]["25%"]
            descriptive[columnName]['Q2:50%'] = dataset.describe()[columnName]["50%"]
            descriptive[columnName]['Q3:75%'] = dataset.describe()[columnName]["75%"]
            descriptive[columnName]['99%'] = np.percentile(dataset[columnName],99)
            descriptive[columnName]['Q4:100%'] = dataset.describe()[columnName]["max"]
            descriptive[columnName]['IQR'] = descriptive[columnName]['Q3:75%'] - descriptive[columnName]['Q1:25%']
            descriptive[columnName]['1.5-Rule'] = 1.5 * descriptive[columnName]['IQR']
            descriptive[columnName]['Lesser_Outlier'] = descriptive[columnName]['Q1:25%'] - descriptive[columnName]['1.5-Rule']
            descriptive[columnName]['Greater_Outlier'] = descriptive[columnName]['Q3:75%'] + descriptive[columnName]['1.5-Rule']
            descriptive[columnName]['Min'] = dataset[columnName].min()
            descriptive[columnName]['Max'] = dataset[columnName].max()
            descriptive[columnName]['Skew'] = dataset[columnName].skew()
            descriptive[columnName]['Kurtosis'] = dataset[columnName].kurtosis()
            descriptive[columnName]['Var'] = dataset[columnName].var()
            descriptive[columnName]['StdD'] = dataset[columnName].std()
        return descriptive     

    
# Finding Outlier columns

    def CheckOutliers(quan):
        Lesser = []
        Greater = []
        for columnName in quan:
            if Descriptive[columnName]['Min'] < Descriptive[columnName]['Lesser_Outlier']:
                Lesser.append(columnName)
            if Descriptive[columnName]['Max'] > Descriptive[columnName]['Greater_Outlier']:
                Greater.append(columnName)
        return Lesser, Greater
       

# Replace Outliers

    def ReplaceOutliers(dataset,columnName):
        for columnName in Lesser:
            dataset[columnName][dataset[columnName]<Descriptive[columnName]['Lesser_Outlier']] = Descriptive[columnName]['Lesser_Outlier']
        
        for columnName in Greater:
            dataset[columnName][dataset[columnName]>Descriptive[columnName]['Greater_Outlier']] = Descriptive[columnName]['Greater_Outlier']
        return ReplaceOutliers

    
# Frequency, Relative Frquency, Cumulative Frequency

    def FreqTable(columnName,dataset):
        FreqTable = pd.DataFrame(columns = ["Unique_Values", "Frequency", "Relative_Frequency", "Culmulative_Frequency"])    
        FreqTable["Unique_Values"] = dataset[columnName].value_counts().index
        FreqTable["Frequency"] = dataset[columnName].value_counts().values
        FreqTable["Relative_Frequency"] = (FreqTable["Frequency"]/103)                          
        FreqTable["Culmulative_Frequency"] = FreqTable["Relative_Frequency"].cumsum()
        return FreqTable

# Probability Density Function

    def get_PDF_Probability(dataset,startrange,endrange):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        
        # kde - Kernal Density Estimation plot (forms the curve)
        # kde_kws - Appearence {'color': 'red', 'linewidth': 2, 'alpha': 0.5 }(for transperency)
        ax = sns.distplot(dataset,kde=True,kde_kws={'color':'blue'},color='Green')
        
        # Drawing range line
        plt.axvline(startrange,color='Red')
        plt.axvline(endrange,color='Red')
        
        # Generate a sample
        sample = dataset
        
        # Calculate Paremeters
        sample_mean = sample.mean()
        sample_stdD = sample.std()
        print("Mean ={:.3f}, Standard Deviation ={:.3f}".format(sample_mean,sample_stdD))
        
        # Define the Distribution
        Dist = norm(sample_mean,sample_stdD)
        
        # Sample Probabilities for a range of outcomes
        values = [value for value in range(startrange,endrange)]
        Probabilities = [Dist.pdf(value) for value in values]         
        # dist.pdf => is a method for obtaining the probability density at given values for continuous distribution.
        Prob = sum(Probabilities)
        print("The Area between the range ({},{}) : {}".format(startrange,endrange,sum(Probabilities)))
        return Prob

# Standard Normal Distribution - Z Score (mean = 0, std = 1)

    def standardND(dataset):
        import seaborn as sns
        
        # Calculate mean and standard deviation    
        mean = dataset.mean()
        stdD = dataset.std()
        
        # Calculate Z-scores    
        Zscore = [((value-mean)/stdD) for value in dataset]
        
        # Plot the distribution of Zscore using distplot    
        sns.distplot(Zscore,kde = True)
        
        # Return the mean of the Z-scores (which should be close to 0)    
        sum(Zscore) / len(Zscore)
    
        
                    
