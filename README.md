# Bachelor Thesis Code Repository

This GitHub repository contains the code developed for my Bachelor's thesis.

## Main Model

The main implementation of the **Quasi Score-Driven (QSD) model using the Barron loss function** can be found in:

`Volatility_and_location_QSDmodel.py`

## Empirical Application

The `Empirical/` folder includes code for a preliminary empirical application of the model. Specifically:

- **Volatility modeling** has been tested using **GOOGLE stock prices**.
- **Location modeling** has been applied to **day-ahead electricity spot prices** in **Denmark (DK)** and neighboring countries.

### Dataset Information

The dataset contains day-ahead electricity spot prices (in DKK) from the **Nord Pool** market. Prices are not updated on weekends or public holidays but are adjusted on the next working day. The market covers Denmark, Norway, Sweden, Finland, and Germany.


 **Note:** The empirical application included in this repository is **not part of the final thesis** and should be considered **preliminary**.
