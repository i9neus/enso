#pragma once

#include <map>
#include "generic/Hash.h"
#include "UIButtonMap.h"

using UIStateID = uint;
using UITriggerHash = uint;
struct UIStateTransition;

enum UIStateTransitionResult : uint
{
    kUIStateOkay = 1,
    kUIStateUpdated = 2,
    kUIStateSucceeded = kUIStateOkay | kUIStateUpdated,

    kUIStateRejected = 4,
    kUIStateError = 0xffffffff
};

enum UIStateTransitionFlags : uint 
{ 
    // These are dummy values. They shouldn't affect the hash so set them to zero    
    kUITriggerOnKeyboard = 0, 
    kUITriggerOnMouseButton = 0,

    kUITriggerOnMouseMove = 1,
    kUITriggerOnMouseWheel = 2
};

struct UIState
{
    enum _attrs : uint { kNull = 0, kInvalid = 0xffffffff };
    
    uint                                              idx;
    std::string                                       id;
    std::function<uint(const UIStateTransition&)>     onExitState;
    std::function<uint(const UIStateTransition&)>     onEnterState;

    uint                                              entryTransitionIdx;
    uint                                              exitTransitionIdx;
};

struct UIStateTransition
{
    enum _attrs : uint { kInvalid = 0xffffffff };
    
    uint                        hash;
    KeyboardButtonMap           keyTrigger;
    MouseButtonMap              mouseTrigger;
    uint                        triggerFlags;
    uint                        sourceStateIdx;
    uint                        targetStateIdx;
    std::function<uint(const UIStateTransition&)> getTargetState;

    inline const bool IsNonDeterministic() const { return getTargetState != nullptr; }
    inline const bool HasDeterministicTarget() const { return targetStateIdx != kInvalid; }
};

class UIStateGraph
{
public:
    UIStateGraph(const KeyboardButtonMap& keyCodes, const MouseButtonMap& mouseCodes) :
        m_keyCodes(keyCodes),
        m_mouseCodes(mouseCodes),
        m_currentState(UIState::kNull)
    {
    }	

    void DeclareState(const std::string& name)
    {
        // Push the new state to the list
        m_uiStateList.push_back(UIState{ uint(m_uiStateList.size()), name, nullptr, nullptr, UIStateTransition::kInvalid, UIStateTransition::kInvalid });
        // Register its index in the map
        m_uiStateMap.emplace(name, m_uiStateList.size() - 1);
    }
    
    template<typename HostType, typename OnEnterState, typename OnLeaveState>
    void DeclareState(const std::string& name, HostType* hostInstance, OnEnterState onEnterState, OnLeaveState onExitState)
    {
        DeclareState(name);        
        m_uiStateList.back().onExitState = std::bind(onExitState, hostInstance, std::placeholders::_1);
        m_uiStateList.back().onEnterState = std::bind(onEnterState, hostInstance, std::placeholders::_1);
    }

    template<typename HostType, typename OnEnterState>
    void DeclareState(const std::string& name, HostType* hostInstance, OnEnterState onEnterState)
    {
        DeclareState(name);
        m_uiStateList.back().onEnterState = std::bind(onEnterState, hostInstance, std::placeholders::_1);
    }

    inline UIState* FindState(const std::string& name)
    { 
        auto it = m_uiStateMap.find(name);
        return (it == m_uiStateMap.end()) ? nullptr : &m_uiStateList[it->second];
    }

    inline const UIState& GetState(const uint idx) const 
    {
        AssertMsg(idx < m_uiStateList.size(), "Invalid state index.");
        return m_uiStateList[idx];
    }

    inline std::string GetTargetStateID(const UIStateTransition& transition) const
    {
        return (transition.targetStateIdx == UIState::kInvalid) ? std::string("") : m_uiStateList[transition.targetStateIdx].id;
    }

    inline void SetState(const std::string& id)
    {
        auto it = m_uiStateMap.find(id);
        AssertMsgFmt(it != m_uiStateMap.end(), "Could not set state to '%s'. State not found.", id.c_str());
        m_currentState = it->second;
    }

    inline void Reset()
    {
        m_currentState = 0;
    }

    void DeclareTransitionImpl(const std::string& sourceStateID, const std::string& targetStateID, const KeyboardButtonMap& keyTrigger, 
                               const MouseButtonMap& mouseTrigger, const uint& triggerFlags)
    {
        UIState* sourceState = FindState(sourceStateID);
        AssertMsgFmt(sourceState, "Error: transition sourceState '%s' was not declared.", sourceStateID.c_str());

        // Each state can only have one exit transition object 
        //AssertMsgFmt(sourceState->exitTransitionIdx == UIStateTransition::kInvalid, "Error: cannot bind transition {'%s' -> '%s'} because state '%s' already has a transition bound to it",
        //    sourceStateID.c_str(), targetStateID.c_str(), sourceStateID.c_str());

        UIState* targetState = nullptr;
        if (!targetStateID.empty())
        {
            targetState = FindState(targetStateID);
            AssertMsgFmt(targetState, "Error: transition targetState '%s' was not declared.", targetStateID.c_str());
        }        
        
        // Generate a hash from the origin state's ID and the trigger required to transition away from it
        const uint hash = HashCombine(uint(std::hash<uint>{}(sourceState->idx)), keyTrigger.HashOf(), mouseTrigger.HashOf(), HashOf(triggerFlags));

        //Log::Error("Hash: 0x%x, 0x%x, 0x%x -> 0x%x", uint(std::hash<uint>{}(0)), keyTrigger.HashOf(), mouseTrigger.HashOf(), hash);

        m_uiTransitionList.push_back(UIStateTransition{ hash, 
                                                        keyTrigger,
                                                        mouseTrigger,
                                                        triggerFlags,
                                                        sourceState->idx, 
                                                        targetState ? targetState->idx : UIState::kInvalid,
                                                        nullptr });

        // Mark the source and target states and being connected up 
        sourceState->exitTransitionIdx = m_uiTransitionList.size() - 1;
        if (targetState)
        {
            targetState->entryTransitionIdx = m_uiTransitionList.size() - 1;
        }

        m_uiStateTransitionMap.emplace(hash, m_uiTransitionList.size() - 1);
    }

    inline void DeclareDeterministicTransition(const std::string& sourceStateID, const std::string& targetStateID, const KeyboardButtonMap& keyTrigger, 
                                               const MouseButtonMap& mouseTrigger, const uint& triggerFlags)
    {
        DeclareTransitionImpl(sourceStateID, targetStateID, keyTrigger, mouseTrigger, triggerFlags);
    }

    template<class HostClass, typename GetTransitionState>
    void DeclareNondeterministicTransition(const std::string& sourceState, const KeyboardButtonMap& keyTrigger, const MouseButtonMap& mouseTrigger, const uint& triggerFlags,
                                           HostClass* hostClass, GetTransitionState getTargetState)
    {
        DeclareTransitionImpl(sourceState, "", keyTrigger, mouseTrigger, triggerFlags);
        m_uiTransitionList.back().getTargetState = std::bind(getTargetState, hostClass, std::placeholders::_1);
    }

	void OnTriggerTransition(const uint triggerFlags)
	{
        const uint hash = HashCombine(uint(std::hash<uint>{}(m_currentState)), m_keyCodes.HashOf(), m_mouseCodes.HashOf(), HashOf(triggerFlags));

        //Log::Error("Hash: 0x%x, 0x%x, 0x%x -> 0x%x", uint(std::hash<uint>{}(0)), m_keyCodes.HashOf(), m_mouseCodes.HashOf(), hash);

		auto range = m_uiStateTransitionMap.equal_range(hash);
		for (auto it = range.first; it != range.second; ++it)
		{
            // If this transition doesn't match the trigger criteria, continue looking
            const auto& transition = m_uiTransitionList[it->second];
            if (transition.sourceStateIdx != m_currentState || transition.keyTrigger != m_keyCodes || 
                transition.mouseTrigger != m_mouseCodes || transition.triggerFlags != triggerFlags) { continue; }
	
            // If the source state has a notification lambda, call it here
            uint sourceStateIdx = m_currentState;
            if (m_uiStateList[m_currentState].onExitState)
            {
                const uint result = m_uiStateList[m_currentState].onExitState(transition);
                AssertMsg(result != kUIStateError, "State transition error.");
                if (result == kUIStateRejected)
                {
                    Log::Warning("Warning: failed to enter state '%s': transition was rejected.",  m_uiStateList[m_currentState].id);
                    break;
                }
            }

            // If the transition lambda attached then it's non-deterministic. Call the lambda to determine what state we're migrating to
            if (transition.IsNonDeterministic())
            {
                m_currentState = transition.getTargetState(transition);
            }
            // Otherwise, treat the transition as deterministic
            else
            {
                m_currentState = transition.targetStateIdx;
            }
            
            // If the target (now the current) state has a lambda, call it now.
            uint targetResult = kUIStateOkay;
            if (m_uiStateList[m_currentState].onEnterState)
            {
                const uint result = m_uiStateList[m_currentState].onEnterState(transition);
                AssertMsg(result != kUIStateError, "State transition error.");

                if (result == kUIStateRejected)
                {
                    Log::Warning("Warning: failed to enter state '%s': transition was rejected.", 
                        transition.IsNonDeterministic() ? std::string("[UNKNOWN]") : m_uiStateList[transition.targetStateIdx].id);
                    break;
                }
            }

            Log::Success("State changed: %s -> %s", m_uiStateList[sourceStateIdx].id, m_uiStateList[m_currentState].id);
            break;
		}
	}

    void Finalise() const
    {
        std::vector<const UIState*> acyclicStates;
        std::vector<const UIState*> orphanedStates;
        for (const auto& state : m_uiStateList)
        {
            if (state.exitTransitionIdx < 0 && state.entryTransitionIdx < 0) { orphanedStates.push_back(&state); }
            else if (state.exitTransitionIdx < 0) { acyclicStates.push_back(&state); }            
        }
        
        Log::Indent indent("Built UI graph:");
        Log::Write("Start state: %s", m_uiStateList[m_currentState].id);
        Log::Write("%i nodes", m_uiStateList.size());
        Log::Write("%i edges", m_uiTransitionList.size());       
        
        if (!acyclicStates.empty())
        {
            Log::Warning("%i acyclic transitions detected! The following nodes are leaves.", acyclicStates.size());
            for (const auto state : acyclicStates) { Log::Warning("  - %s", state->id); }
        }
        else
        {
            Log::Success("No acyclic walks!");
        }

        if (!orphanedStates.empty())
        {
            Log::Warning("%i orphaned nodes detected! The following have no transitions in or out.", orphanedStates.size());
            for (const auto state : orphanedStates) { Log::Warning("  - %s", state->id); }
        }
        else
        {
            Log::Success("No orphaned states!");
        }
    }

private:
    std::vector<UIState>                            m_uiStateList;
    std::map<std::string, uint>                     m_uiStateMap;
    std::vector<UIStateTransition>                  m_uiTransitionList;
    std::multimap<uint, uint>                       m_uiStateTransitionMap;

	const KeyboardButtonMap&                        m_keyCodes;
	const MouseButtonMap&   						m_mouseCodes;

    uint                                            m_currentState;
};