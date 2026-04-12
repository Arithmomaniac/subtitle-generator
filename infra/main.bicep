targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment (used as prefix for all resources)')
param environmentName string = 'subtitlegen'

@description('Azure region for all resources')
param location string = 'eastus'

@description('Principal ID of the signed-in user (for Storage RBAC during dev). Optional.')
param principalId string = ''

@description('Email address for alert notifications')
param alertEmail string = ''

var resourceGroupName = '${environmentName}-rg'
var storageAccountName = replace(toLower('${environmentName}st'), '-', '')
var functionAppName = '${environmentName}-func'
var functionPlanName = '${environmentName}-plan'
var logAnalyticsName = '${environmentName}-logs'
var appInsightsName = '${environmentName}-ai'

resource rg 'Microsoft.Resources/resourceGroups@2024-11-01' = {
  name: resourceGroupName
  location: location
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    storageAccountName: storageAccountName
    location: location
    principalId: principalId
  }
}

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  scope: rg
  params: {
    logAnalyticsName: logAnalyticsName
    appInsightsName: appInsightsName
    location: location
  }
}

module functionApp 'modules/functionapp.bicep' = {
  name: 'functionapp'
  scope: rg
  params: {
    functionAppName: functionAppName
    functionPlanName: functionPlanName
    location: location
    storageAccountName: storage.outputs.storageAccountName
    appInsightsConnectionString: monitoring.outputs.appInsightsConnectionString
    staticWebsiteUrl: storage.outputs.staticWebsiteUrl
  }
}

module alerts 'modules/alerts.bicep' = {
  name: 'alerts'
  scope: rg
  params: {
    appInsightsId: monitoring.outputs.appInsightsId
    location: location
    alertEmail: alertEmail
    functionAppName: functionAppName
  }
}

output functionAppName string = functionApp.outputs.functionAppName
output storageAccountName string = storage.outputs.storageAccountName
output staticWebsiteUrl string = storage.outputs.staticWebsiteUrl
output resourceGroupName string = resourceGroupName
