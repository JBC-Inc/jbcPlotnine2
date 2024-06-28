#install.packages("reticulate")
library(reticulate)

# Run once ======================================================================
# reticulate::install_python(version = "3.12.3", force = TRUE)
#
# Virtual Environments are created from another "starter" or "seed" Python already installed on the system.
# Suitable Pythons installed on the system are found by
reticulate::virtualenv_starter()
#
# Create interface to Python Virtual Environment
reticulate::virtualenv_create(
  force = TRUE,
  envname = "python-env",
  version = "3.12.3"
  )
#
# Install packages (once)
# reticulate::virtualenv_install(envname = "python-env", packages = c("datetime"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("patchworklib"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("shiny"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("shinyswatch"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("shinywidgets"))
# reticulate::virtualenv_install(envname = "python-env", packages = c("plotnine"))

# lookat information about the version of Python currently being used by reticulate.
# reticulate::py_config()

# Select the version of Python to be used by reticulate
reticulate::use_virtualenv(
  virtualenv = "python-env",
  required = TRUE
)

# Create interactive Python console within R.
# Objects created within Python are available for R session (and vice-versa).
reticulate::repl_python()




# <('.'<)  <('.')>  (>'.')>


























