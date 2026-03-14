package messaging

const (
	SubjectMentionsRaw      = "mentions.raw"
	SubjectMentionsFiltered = "mentions.filtered"
	SubjectMentionsEnriched = "mentions.enriched"
	SubjectMentionsReady    = "mentions.ready"
	SubjectAlertsTrigger    = "alerts.trigger"
	SubjectSourcesChanged   = "sources.changed"

	StreamMentions = "MENTIONS"
	StreamAlerts   = "ALERTS"
	StreamSources  = "SOURCES"
)
