import sys
from time import gmtime, strftime
import pickle


class Logger:
    def __init__(self, add_timestamp=True, logfile_name=None, logfile_name_time_suffix=True, print_to_screen=True):
        self.add_timestamp = add_timestamp
        self.logfile = logfile_name
        self.logfile_basename = logfile_name

        self.print_to_screen = print_to_screen

        if self.logfile:
            # Remove ".txt" postfix (it will be added later)
            if self.logfile[-4:] == ".txt":
                self.logfile = self.logfile[0:-4]

            # Add timestamp suffix
            if logfile_name_time_suffix:
                self.logfile += strftime("-%m-%d_%H-%M-%S", gmtime())
            self.logfile_basename = self.logfile
            self.logfile += ".txt"

            try:
                f = open(self.logfile, "a")
                f.close()
            except FileNotFoundError:
                print("Could not open logfile, quitting.")
                sys.exit(-1)

    def save_variables(self, var, var_name=""):
        if not self.logfile_basename:
            raise NotImplementedError
        with open(self.logfile_basename + "." + var_name + ".pkl", 'wb') as f:
            pickle.dump(var, f)

    def _do_print(self, message, type_):
        message = message
        if self.add_timestamp:
            message = strftime("[%Y-%m-%d %H:%M:%S " + type_ + "] ", gmtime()) + message
        if self.print_to_screen:
            print(message)
        if self.logfile:
            with open(self.logfile, "a") as logfile:
                logfile.write(message + "\n")

    def get_log_basename(self):
        return self.logfile_basename

    def create_desc_file(self, desc, postfix=".desc.txt"):
        try:
            with open(self.logfile_basename + postfix, "a") as descfile:
                descfile.write(desc)
        except FileNotFoundError:
            print("Could not create desc file (" + str(self.logfile_basename + postfix) + ")! Quitting")
            sys.exit(-1)
        return self.logfile_basename

    def info(self, message):
        self._do_print(message, "Info")

    def stats(self, message):
        self._do_print(message, "Stats")
