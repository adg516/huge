import matplotlib.pyplot as plt

# Read costs from file
with open('costs.txt', 'r') as f:
    costs = [float(line.strip()) for line in f]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(costs, color='blue')
plt.title('Cost per Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid(True)
plt.show()
