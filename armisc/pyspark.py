"""
Creates a new connection to spark and makes available: 
`spark`, `sq` (`SQLContext`), `F`, and `Window` in the global namespace.
"""

def _parse_master(pyspark_submit_args):
    sargs = pyspark_submit_args.split()
    for j, sarg in enumerate(sargs):
        if sarg == "--master":
            try:
                return sargs[j + 1]
            except:
                raise Exception("Could not parse master from PYSPARK_SUBMIT_ARGS")
    raise Exception("Could not parse master from PYSPARK_SUBMIT_ARGS")


def initialize_spark(appName="MyApp", submit_args=None, memory=12):
    """
    This function assumes you already have SPARK_HOME and PYSPARK_SUBMIT_ARGS environment variables set
    """
    import os
    import findspark
    from textwrap import dedent

    if "SPARK_HOME" not in os.environ:
        raise Exception("SPARK_HOME environmental variable not set.")
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        os.environ[
            "PYSPARK_SUBMIT_ARGS"
        ] = f"--master local[12] --driver-memory {memory}g --executor-memory {memory}g pyspark-shell"
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        raise Exception(
            dedent(
                """\
                       PYSPARK_SUNBMIT_ARGS environmental variable not set.
                       
                       As an example:
                       export PYSPARK_SUBMIT_ARGS = " --master local[8] --driver-memory 8g --executor-memory 8g pyspark-shell"
                       """
            )
        )
    findspark.init(os.environ["SPARK_HOME"])
    spark_master = _parse_master(os.environ["PYSPARK_SUBMIT_ARGS"])
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master(spark_master).appName("MyApp").getOrCreate()
    return spark


def assert_pyspark():
    import pyspark.sql.functions as F
    from pyspark.sql import Window

    return F, Window


def load_spark():
    spark = initialize_spark()
    return spark
