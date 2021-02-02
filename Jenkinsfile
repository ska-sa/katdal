#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.setDependencies([
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katpoint/master',
    'ska-sa/katsdptelstate/master'])
katsdp.standardBuild()
katsdp.mail('ludwig@ska.ac.za')
