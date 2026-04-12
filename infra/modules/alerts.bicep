@description('Application Insights resource ID')
param appInsightsId string

@description('Azure region')
param location string

@description('Email address for alert notifications')
param alertEmail string

@description('Function App name (for alert naming)')
param functionAppName string

resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: '${functionAppName}-alerts-ag'
  location: 'global'
  properties: {
    groupShortName: 'SubtGen'
    enabled: true
    emailReceivers: [
      {
        name: 'SubtitleGenAdmin'
        emailAddress: alertEmail
        useCommonAlertSchema: false
      }
    ]
  }
}

// Alert: Function execution failures
resource functionFailureAlert 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: '${functionAppName}-function-failure'
  location: location
  properties: {
    displayName: 'Subtitle Gen: Function execution failed'
    description: 'A subtitle-generator function invocation failed. Check App Insights for details.'
    severity: 1
    enabled: true
    evaluationFrequency: 'PT5M'
    windowSize: 'PT5M'
    scopes: [appInsightsId]
    criteria: {
      allOf: [
        {
          query: 'requests | where success == false | project timestamp, name, resultCode, duration, id | summarize failureCount = count(), functions = strcat_array(make_set(name, 10), \', \'), lastResultCode = take_any(resultCode) by bin(timestamp, 5m)'
          timeAggregation: 'Count'
          operator: 'GreaterThan'
          threshold: 0
          dimensions: [
            { name: 'functions', operator: 'Include', values: ['*'] }
            { name: 'lastResultCode', operator: 'Include', values: ['*'] }
          ]
          failingPeriods: {
            numberOfEvaluationPeriods: 3
            minFailingPeriodsToAlert: 2
          }
        }
      ]
    }
    autoMitigate: true
    actions: {
      actionGroups: [actionGroup.id]
    }
  }
}

// Alert: Errors logged (severityLevel >= 3 = Error+)
resource errorLogAlert 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: '${functionAppName}-error-logs'
  location: location
  properties: {
    displayName: 'Subtitle Gen: Errors in application logs'
    description: 'Error-level log entries appeared in traces — may indicate storage or processing failures.'
    severity: 2
    enabled: true
    evaluationFrequency: 'PT15M'
    windowSize: 'PT15M'
    scopes: [appInsightsId]
    criteria: {
      allOf: [
        {
          query: 'traces | where severityLevel >= 3 | project timestamp, message = substring(message, 0, 200) | summarize errorCount = count(), errors = strcat_array(make_set(message, 10), \' | \') by bin(timestamp, 15m)'
          timeAggregation: 'Count'
          operator: 'GreaterThan'
          threshold: 0
          dimensions: [
            { name: 'errors', operator: 'Include', values: ['*'] }
          ]
          failingPeriods: {
            numberOfEvaluationPeriods: 3
            minFailingPeriodsToAlert: 2
          }
        }
      ]
    }
    autoMitigate: true
    actions: {
      actionGroups: [actionGroup.id]
    }
  }
}

// Alert: Client-side (browser) exceptions
resource clientExceptionAlert 'Microsoft.Insights/scheduledQueryRules@2023-03-15-preview' = {
  name: '${functionAppName}-client-exceptions'
  location: location
  properties: {
    displayName: 'Subtitle Gen: Client-side exceptions'
    description: 'Browser-side JavaScript exceptions detected. Check App Insights exceptions table for stack traces.'
    severity: 2
    enabled: true
    evaluationFrequency: 'PT15M'
    windowSize: 'PT15M'
    scopes: [appInsightsId]
    criteria: {
      allOf: [
        {
          query: 'exceptions | where client_Type == "Browser" | project timestamp, type, outerMessage = substring(outerMessage, 0, 200) | summarize exceptionCount = count(), types = strcat_array(make_set(type, 10), \', \') by bin(timestamp, 15m)'
          timeAggregation: 'Count'
          operator: 'GreaterThan'
          threshold: 0
          dimensions: [
            { name: 'types', operator: 'Include', values: ['*'] }
          ]
          failingPeriods: {
            numberOfEvaluationPeriods: 3
            minFailingPeriodsToAlert: 2
          }
        }
      ]
    }
    autoMitigate: true
    actions: {
      actionGroups: [actionGroup.id]
    }
  }
}
