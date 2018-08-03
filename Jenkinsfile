#!groovy

@Library('katsdpjenkins') _
katsdp.setDependencies([
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katpoint/master',
    'ska-sa/katsdptelstate/master'])
katsdp.standardBuild(python3: true)
katsdp.mail('ludwig@ska.ac.za')
