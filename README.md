<hr>
# CP3 Linear least square
 
## Ideas for this project:
* Predicting potential of a football U23 player, based on popular video game FIFA. Considering age, workout hours per week and current overall rating.
* Predicting car condition over years. Same logic, now considering age of the car and how many times does it get checked by technician per year

## Generating artificial data
> I generated players and cars(brands and owners) artificially
> However, program still can work with real data

## Implementation:
### General:
  I created corresponding classes with attributes: 
    player with full name, age, workout hours, overall rating and potential; 
    cars with owner, brand, age, checkings per year and condition (let's say every car starts at peak condition --> 100.00 and then decreases)
  Equations for estimating their potential/condition is made by me.
  After that I generated training data and created matrix from their properties.
  
  If you run this program you will get a prediction of a custom object and after that see generated predictions for new testing data (about 50 entries)
    

### Differing Info:
 For the first idea, I use matrix of people to solve (Ax = b) x by normal equation,modified Gram-Schmidt or Householder depending on user's input.
 
 For the second idea, I use matrix of cars to solve (Ax = b) with custom constraint (Cx = d) by modified Gram-Schmidt or Householder depending on user's input.

<hr/>
