import ast
import matplotlib.pyplot as plt

list_of_keys_to_dot = ["roll_output", "pitch_output", "yaw_output", "depth_output",]

def read_and_process_file(file_path):
    list_of_dicts = []
    with open(file_path, "r") as file:
        # print the keys of the first line
        for line in file:
            data_dict = ast.literal_eval(line.strip())
            print(data_dict.keys())
            break

        for line in file:
            data_dict = ast.literal_eval(line.strip())
            # store the dictionary in a list
            list_of_dicts.append(data_dict)

    return list_of_dicts


def plot_data(data, list_to_plot):
    read_dict = {}
    obstacle_interrupt = [data_dict["obstacle_interrupt"] for data_dict in data]

    for key in list_to_plot:
        read_dict[key] = [data_dict[key] for data_dict in data]

        if key in list_of_keys_to_dot:
            plt.plot(read_dict[key], label="elevons angle", linestyle='--')
        else:
            if key == "roll":
                plt.plot(read_dict[key], label="vehicle roll angle", linestyle='-')
            elif key == "pitch":
                plt.plot(read_dict[key], label="vehicle pitch angle", linestyle='-')

    # create a line at the bottom if the obstacle direction is 'left'
    for i, [there_is, direction] in enumerate(obstacle_interrupt):
        if there_is:
            if direction == "left":
                plt.axvline(x=i, ymin=0, ymax=0.05, color="red", linestyle="-")
            elif direction == "right":
                plt.axvline(x=i, ymin=0, ymax=0.05, color="blue", linestyle="-")
            elif direction == "up":
                plt.axvline(x=i, ymin=0, ymax=0.05, color="green", linestyle="-")
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Angle')
    plt.title('sensor data vs time')
    plt.show()

# Replace 'your_file_path.txt' with the actual path to your text file
data = read_and_process_file("vlogs/2024-05-27_19-49-32_stats.txt")

plot_data(data, ["roll", "pitch"])

# TODO attach the plot to the video and make a moving line to indicate when it is in the water and when it is not