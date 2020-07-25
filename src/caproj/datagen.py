"""
caproj.datagen
~~~~~~~~~~~~~~

This module contains functions for generating the interval metrics used in modeling
for each unique capital project

**Module variables:**

.. autosummary::

   endstate_columns
   endstate_column_rename_dict
   info_columns
   info_column_rename_dict

**Module functions:**

.. autosummary::

   print_record_project_count
   generate_interval_data
   print_interval_dict

"""

import os

import pandas as pd

#: List of column names containing info for each project's end-state
endstate_columns = [
    "Date_Reported_As_Of",
    "Change_Years",
    "PID",
    "Current_Phase",
    "Budget_Forecast",
    "Forecast_Completion",
    "PID_Index",
]

#: Dictionary for mapping members of ``endstate_columns`` to new column names
endstate_column_rename_dict = {
    "Date_Reported_As_Of": "Final_Change_Date",
    "Current_Phase": "Phase_End",
    "Budget_Forecast": "Budget_End",
    "Forecast_Completion": "Schedule_End",
    "PID_Index": "Number_Changes",
    "Change_Years": "Final_Change_Years",
}

#: List of column names containing descriptive info for each project
info_columns = [
    "PID",
    "Project_Name",
    "Description",
    "Category",
    "Borough",
    "Managing_Agency",
    "Client_Agency",
    "Current_Phase",
    "Current_Project_Years",
    "Current_Project_Year",
    "Design_Start",
    "Original_Budget",
    "Original_Schedule",
]

#: Dictionary for mapping members of ``info_columns`` to new column names
info_column_rename_dict = {
    "Current_Phase": "Phase_Start",
    "Original_Budget": "Budget_Start",
    "Original_Schedule": "Schedule_Start",
}


def print_record_project_count(dataframe, dataset="full"):
    """Prints summary of records and unique projects in dataframe

    :param dataframe: pd.DataFrame object for the version of the NYC capital
                      projects data you wish to summarize
    :param dataset: string, accepts 'full', 'all', 'training', or 'test'
                    (default 'full')

    :return: prints to standard output, no objects returned
    """
    if dataset == "full":
        print(
            "For the ORIGINAL cleansed data, containing all available NYC capital "
            "projects change records:\n"
        )

    elif dataset == "all":
        print(
            "For the data containing start and end data for all available "
            "NYC capital projects for the ENTIRE INTERVAL of changes "
            "covered in the ORIGINAL data:\n"
        )

    else:
        print(
            "For the final {} data, containing the {} split of 3-year "
            "project data used in this analysis:\n".format(
                dataset.upper(), dataset
            )
        )

    # entries
    print(f"\tNumber of dataset records: {len(dataframe)}")

    # num projects
    print(
        f"\tNumber of unique projects in dataset: {dataframe['PID'].nunique()}\n"
    )


# define the functions used for generating our interval dataframe


def ensure_datetime_and_sort(df):
    """Ensures datetime columns are formatted correctly and changes are sorted

    :param df: pd.DataFrame of the cleaned capital projects change records data

    :return: Original pd.DataFrame with datetime columns formatted and records
             sorted
    """
    datetime_cols = [
        "Date_Reported_As_Of",
        "Design_Start",
        "Original_Schedule",
        "Forecast_Completion",
    ]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])

    # make sure data is sorted properly
    df = df.sort_values(by=["PID", "PID_Index"])

    return df


def extract_project_details(
    df,
    copy_columns=info_columns,
    column_rename_dict=info_column_rename_dict,
    use_record=0,
    record_index="PID_Index",
):
    """Generates a dataframe with project details for each unique PID

    :param df: The cleaned capital projects change records data
    :type df: dataframe
    :param copy_columns: list of the names of columns that should be copied
                         containing primary information about each project,
                         defaults to info_columns
    :type copy_columns: list, optional
    :param column_rename_dict: dict of column name mappings to rename copied
                               columns, defaults to info_column_rename_dict
    :type column_rename_dict: dict, optional
    :param use_record: integer record_index value to use as the basis the
                       resulting project info, defaults to 0 (indicating that
                       the first chronological record for each project will be used)
    :type use_record: int, optional
    :param record_index: indicates the column name to use for the
                         record_index referenced use_record, defaults to "PID_Index"
    :type record_index: str, optional
    :return: dataframe containing the primary project details for each
             unique PID, and the PID is set as the index
    :rtype: dataframe
    """
    df_details = df.copy().loc[df[record_index] == use_record][copy_columns]

    if column_rename_dict:
        df_details = df_details.copy().rename(columns=column_rename_dict)

    return df_details.set_index("PID")


def subset_project_changes(
    df,
    change_year_interval=3,
    change_col="Change_Year",
    project_age_col="Current_Project_Year",
    inclusive_stop=True,
):
    """Generates a subsetted dataframe with only the change records that occur
    in or before the specified max interval year

    :param df: pd.DataFrame of the cleaned capital projects change records data
    :param change_year_interval: integer representing the maximum year from which
                                 to include changes for each project (default
                                 change_year_interval=3)
    :param change_col: string, name of column containing change year indicators
                       (default change_col='Change_Year')
    :param project_age_col: string, name of column containing current age of
                            each project at the time the dataset was compiled
                            (default project_age_col='Current_Project_Year')
    :param inclusive_stop: boolean, indicating whether projects to be included
                           in the subset dataframe need to be older than the
                           change_year_interval year or can be
                           equal-to-or-older-than the change_year_interval year
                           If True, >= is used for subsetting, if False > is
                           used (default inclusive_stop=True)

    :return: pd.DataFrame of the subsetted data, the index is set to each
             record's 'Record_ID' value
    """
    df_subset = df.copy().loc[
        (df[change_col] <= change_year_interval)
        & (
            df[project_age_col] >= change_year_interval
            if inclusive_stop
            else df[project_age_col] > change_year_interval
        )
    ]

    return df_subset.set_index("Record_ID")


def find_max_record_indices(df, record_index="PID_Index"):
    """Creates a list of Record_ID values of the max record ID for each PID

    :param df: pd.DataFrame containing the cleaned capital project change
               records
    :param record_index: string name of column containing PID ordinal
                         indices (defaul record_index='PID_Index')

    :return: list of max Record_ID values for each PID
    """
    df_group = df.groupby("PID").agg({record_index: max})
    pid_dict = dict(zip(df_group.index, df_group.values.ravel()))
    record_id_indices = [
        str(pid) + "-" + str(pid_index) for pid, pid_index in pid_dict.items()
    ]

    return record_id_indices


def project_interval_endstate(
    df,
    keep_columns=endstate_columns,
    column_rename_dict=endstate_column_rename_dict,
    change_year_interval=None,
    record_index="PID_Index",
    change_col="Change_Year",
    project_age_col="Current_Project_Year",
    inclusive_stop=True,
):
    """Generates a dataframe of endstate data for each unique PID given the
    specified analysis interval

    :param df: pd.DataFrame of the cleaned capital projects change records data
    :param keep_columns: list of column names for columns that should be kept
                         as part of the resulting dataframe (default
                         keep_columns=endstate_columns module variable)
    :param column_rename_dict: dict mapping existing column names to the new
                               names to which they should be named (default
                               column_rename_dict=endstate_column_rename_dict
                               module variable)
    :param change_year_interval: integer or None representing the maximum year
                                 from which to include changes for each
                                 project,  if None, then all years' worth of
                                 changes included (default
                                 change_year_interval=None)
    :param record_index: string name of column containing PID ordinal
                         indices (defaul record_index='PID_Index')
    :param change_col: string, name of column containing change year indicators
                       (default change_col='Change_Year')
    :param project_age_col: string, name of column containing current age of
                            each project at the time the dataset was compiled
                            (default project_age_col='Current_Project_Year')
    :param inclusive_stop: boolean, indicating whether projects to be included
                           in the subset dataframe need to be older than the
                           change_year_interval year or can be
                           equal-to-or-older-than the change_year_interval year
                           If True, >= is used for subsetting, if False > is
                           used (default inclusive_stop=True)

    :return: pd.DataFrame containing endstate data for each unique project,
             the index is set to the PID
    """
    if change_year_interval:
        df = subset_project_changes(
            df.copy(),
            change_year_interval,
            change_col,
            project_age_col,
            inclusive_stop,
        )
    else:
        df = df.copy().set_index("Record_ID")

    max_record_list = find_max_record_indices(df, record_index)

    df_endstate = df.copy().loc[max_record_list][keep_columns]

    if column_rename_dict:
        df_endstate = df_endstate.copy().rename(columns=column_rename_dict)

    return df_endstate.set_index("PID")


def join_data_endstate(df_details, df_endstate, how="inner"):
    """Creates dataframe joining the df_details and df_endstate dataframes by PID

    :param df_details: pd.DataFrame output from the extract_project_details()
                       function
    :param df_endstate: pd.DataFrame output from the project_interval_endstate()
                        function
    :param how: string passed to the pd.merge method indicating the type
                of join to perform (default how='inner')

    :return: pd.DataFrame containing the join results, the index is reset
    """
    df_join = pd.merge(
        df_details, df_endstate, how=how, left_index=True, right_index=True
    )

    return df_join.reset_index()


def add_change_features(df):
    """Calculates interval change metrics for each PID and appends the dataset

    :param df: pd.DataFrame containing joined project interval data output
               from the join_data_endstate() function

    :return: Copy of input pd.DataFrame with the new metrics appended as
             additional columns
    """
    # copy input for comparison of outputs
    df_copy = df.copy()

    # calculate interval change features
    df_copy["Duration_Start"] = (
        df_copy["Schedule_Start"] - df_copy["Design_Start"]
    ).dt.days
    df_copy["Duration_End"] = (
        df_copy["Schedule_End"] - df_copy["Design_Start"]
    ).dt.days
    df_copy["Schedule_Change"] = (
        df_copy["Duration_End"] - df_copy["Duration_Start"]
    )
    df_copy["Budget_Change"] = df_copy["Budget_End"] - df_copy["Budget_Start"]

    # define schedule change ratio
    df_copy["Schedule_Change_Ratio"] = (
        df_copy["Schedule_Change"] / df_copy["Duration_Start"]
    )
    # define budget change ratio
    df_copy["Budget_Change_Ratio"] = (
        df_copy["Budget_Change"] / df_copy["Budget_Start"]
    )

    # define project metrics
    df_copy["Budget_Abs_Per_Error"] = (
        df_copy["Budget_Start"] - df_copy["Budget_End"]
    ).abs() / df_copy["Budget_End"]

    df_copy["Budget_Rel_Per_Error"] = (
        df_copy["Budget_Start"] - df_copy["Budget_End"]
    ).abs() / df_copy["Budget_Start"]

    df_copy["Duration_End_Ratio"] = (
        df_copy["Duration_End"] / df_copy["Duration_Start"]
    )
    df_copy["Budget_End_Ratio"] = (
        df_copy["Budget_End"] / df_copy["Budget_Start"]
    )

    # previously titled 'Mark Metric'
    df_copy["Duration_Ratio_Inv"] = (
        df_copy["Duration_Start"] / df_copy["Duration_End"]
    ) - 1
    df_copy["Budget_Ratio_Inv"] = (
        df_copy["Budget_Start"] / df_copy["Budget_End"]
    ) - 1

    return df_copy


def generate_interval_data(
    data,
    change_year_interval=None,
    inclusive_stop=True,
    to_csv=False,
    save_dir="../data/interim/",
    custom_filename=None,
    verbose=1,
    return_df=True,
):
    """Generates a project analysis dataset for the specified interval

    NOTE:

        If you specify ``to_csv=True``, the default bahavior will be to
        save the resulting dataframe as:

        ``../data/interim/NYC_capital_projects_{predict_interval}yr.csv``

        or if ``change_year_interval=None``:

        ``../data/interim/NYC_capital_projects_all.csv``

        The ``save_dir`` and ``custom_filename`` arguments allow you to change
        this ``to_csv`` behavior, however using them is not recommended
        for the sake of file naming consistency in this project.

    :param data: pd.DataFrame of the cleaned capital projects change
                 records data
    :param change_year_interval: integer or None representing the maximum year
                                 from which to include changes for each
                                 project,  if None, then all years' worth of
                                 changes included (default
                                 change_year_interval=None)
    :param inclusive_stop: boolean, indicating whether projects to be included
                           in the subset dataframe need to be older than the
                           change_year_interval year or can be
                           equal-to-or-older-than the change_year_interval year
                           If True, >= is used for subsetting, if False > is
                           used (default inclusive_stop=True)
    :param to_csv: boolean, indicating whether or not the resulting dataframe
                   should be saved to disk (default to_csv=False)
    :param save_path: string or None, indicating the path to which the
                      resulting dataframe should be saved to .csv, if None
                      the dataframe is not saved, just returned (default
                      save_path=None)
    :param custom_filename: string or None, indicating whether to name the
                            resulting .csv file something other than the
                            name 'NYC_capital_projects_{interval}yr.csv'
                            (default custom_filename=None)
    :param verbose: integer, default verbose=1 prints the number of project
                    remaining in the resulting dataframe, otherwise that
                    information is not printed
    :param return_df: boolean, determines whether the resulting pd.DataFrame
                      object is returned (default return_df=True)

    :return: pd.DataFrame containing the summary change data for each unique
             project matching the specified change_year_interval
    """
    data = ensure_datetime_and_sort(data.copy())

    df_endstate = project_interval_endstate(
        data,
        change_year_interval=change_year_interval,
        inclusive_stop=inclusive_stop,
    )

    df_details = extract_project_details(data)

    df_merged = join_data_endstate(df_details, df_endstate)

    df_features = add_change_features(df_merged)

    if verbose == 1:
        # print numbeer of projects in the resulting dataframe
        print(
            "The number of unique projects in the resulting dataframe: {}\n"
            "".format(df_features["PID"].nunique())
        )

    if to_csv:
        if custom_filename:
            save_path = os.path.join(save_dir, custom_filename)
        else:
            filename_base = "NYC_capital_projects_"
            if change_year_interval:
                save_path = os.path.join(
                    save_dir,
                    "{}{}yr.csv".format(filename_base, change_year_interval),
                )
            else:
                save_path = os.path.join(
                    save_dir, "{}all.csv".format(filename_base)
                )

        df_features.to_csv(save_path, index=False)

        print(
            "The resulting interval features dataframe was saved to .csv at:"
            "\n\n\t{}\n".format(save_path)
        )

    if return_df:
        return df_features


def print_interval_dict(
    datadict_dir="../references/data_dicts/",
    datadict_filename="data_dict_interval.csv",
):
    """Prints summary of data dictionary for the generate_interval_data output

    :param datadict_dir: optional string indicating directory location of
                         target data dictionary (default
                         ``../references/data_dicts/``)
    :param datadict_filename: optional string indicating filename of target
                              data dict (default ``data_dict_interval.csv``)

    :return: No objects are returned, printed output only
    """
    filepath = os.path.join(datadict_dir, datadict_filename)

    data_dict = pd.read_csv(filepath)

    print("DATA DICTIONARY: GENERATED INTERVAL DATASETS\n")

    for i, (name, descr, datatype) in enumerate(
        zip(
            data_dict["name"].values,
            data_dict["description"].values,
            data_dict["dtype"].values,
        )
    ):
        print("{}: {} ({})\n\n\t{}\n\n".format(i, name, datatype, descr))
