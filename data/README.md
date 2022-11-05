Data for the ML project should be stored separately from the codebase. Also, ideally you wouldn't want to mix up data from different sources, raw data with processed data etc. To accomodate this, in this folder we can have the following structure:

**Obligatory**:
> 1. `raw` - folder with immutable raw data

**Optional**:
> 1. `external` - raw data from external sources.
> 2. `processed` - data that has been already preprocessed.
> 3. `interim` - intermediate data that has already been transformed. 